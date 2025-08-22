import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


# 启动命令：
# uv run client.py
# 客户端中会同时启动server.py并通过标准输入输出建立连接
# 在窗口中输入问题后
# 1. 挑选模板，拉取相关资源信息补充到模板中
# 2. 生成工具
# 3. 调用大模型Api，获取结果，并且结果中是否有工具需要调用，如果有则处理调用工具，将工具结果补全到上下文中继续调用大模型Api
# 4. 将结果输出到窗口

class MCPClient:
    """MCP客户端，用于与OpenAI API交互并调用MCP工具"""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.mcp_session: Optional[ClientSession] = None
        self.model = "Qwen3-235B"
        self.ai_client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                base_url="http://chat.qwen.com")

    async def connect_to_mcp_server(self):

        # 配置服务端参数
        server_params = StdioServerParameters(command="python", args=["server.py"], env=None)

        # 启动服务端子进程并建立标准输入/输出通信通道，管理进程生命周期
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))  # stdio_client创建一个异步上下文管理器，负责启动服务端进程

        # 获取服务端读取和写入接口
        self.stdio, self.write = stdio_transport

        # 创建会话，负责处理MCP协议通信、消息序列化/反序列化，管理会话生命周期
        self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # 初始化会话
        await self.mcp_session.initialize()

    async def print_mcp_resources(self):

        print(f"\nMCP服务端资源明细:")

        tools_response = await self.mcp_session.list_tools()
        for tool in tools_response.tools:
            print(f"    ====> 工具名称: {tool.name}, 描述: {tool.description}")

        resources_response = await self.mcp_session.list_resources()
        for resource in resources_response.resources:
            print(f"    ====> 资源名称: {resource.uri}, 描述: {resource.description}")

        resources_templates_response = await self.mcp_session.list_resource_templates()
        for resource in resources_templates_response.resourceTemplates:
            print(f"    ====> 资源名称: {resource.uriTemplate}, 描述: {resource.description}")

        prompts_response = await self.mcp_session.list_prompts()
        for prompt in prompts_response.prompts:
            print(f"    ====> Prompt名称: {prompt.name}, 描述: {prompt.description}")

    async def select_prompt_and_enhanced_query(self, query: str) -> str:
        """简单根据问题选择模板，实际可能需要使用语义空间邻近算法"""
        isBJ = any(indicator in query for indicator in ["北京"])
        isTourist = any(indicator in query for indicator in ["景点", "旅游"])

        # 这里只处理北京旅游的提示模板，其它情况不处理
        if not (isBJ and isTourist):
            return query

        # 获取资源
        resources_uri = "city-tourist-doc://北京/city-tourist.md"
        resource = await self.mcp_session.read_resource(resources_uri)
        city_info = resource.contents[0].text

        # 获取模板
        template_name = "top_beijing_tourist_spots"
        prompt_response = await self.mcp_session.get_prompt(template_name, arguments={"city_info": city_info})
        prompt_text = prompt_response.messages[0].content.text

        # 增强提示，将模板内容和用户问题拼接
        enhanced_query = f"{prompt_text}\n\n用户问题:\n\n{query}"
        print("\n[增强提示: ", enhanced_query, "]")
        return enhanced_query

    async def tools_to_openai_functions(self) -> List[Dict]:
        """将mcp中的tools转换为OpenAI函数"""
        tools_response = await self.mcp_session.list_tools()
        functions = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema or {}
            }
        } for tool in tools_response.tools]
        return functions

    async def process_query(self, query: str) -> str:
        """处理用户查询并调用必要的工具"""
        if not self.mcp_session:
            return "❌ 未连接到MCP服务器"

        try:
            # 1. 根据输入的问题选择合适的提示模板进行增强提示
            enhanced_query = await self.select_prompt_and_enhanced_query(query)

            # 2. 将工具列表转换为OpenAI函数格式
            openai_tools = await self.tools_to_openai_functions()

            # 3. 发送初始请求到OpenAI
            messages = [{"role": "user", "content": enhanced_query}]
            response = self.ai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools
            )
            assistant_output = response.choices[0].message
            messages.append(assistant_output.model_dump())
            # print(f"\n[首次请求响应: {assistant_output}]")

            # 判断响应中是否包含工具调用
            while assistant_output.tool_calls:

                # 将所有模型提示调用的工具处理一遍
                for tool_call in assistant_output.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # 调用工具
                    print(f"\n[正在调用工具 {tool_name}, 参数: {tool_args}]")
                    result = await self.mcp_session.call_tool(tool_name, tool_args)
                    print(f"\n[调用工具成功 {tool_name}, 结果: {result}]")

                    # 添加工具响应到消息历史
                    messages.append({
                        "role": "tool",
                        "content": result.content[0].text if hasattr(result, 'content') else str(result),
                        "tool_call_id": tool_call.id,
                    })

                # 再次请求OpenAI
                # print(f"\n[再次请求参数: {messages}]")
                response = self.ai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools
                )
                assistant_output = response.choices[0].message
                messages.append(assistant_output.model_dump())
                # print(f"\n[再次请求响应: {assistant_output}]")

            # 7. 返回最终结果
            return assistant_output.content

        except Exception as e:
            return f"❌ 处理查询时出错: {str(e)}"

    async def chat_loop(self):
        print("\n  开始对话: 输入'/q'退出")

        while True:
            try:
                query = input("\n>>>: ").strip()
                if query == "":
                    continue
                if query.lower() == "/q":
                    break

                print("\n  处理中...")
                response = await self.process_query(query)
                print(f"\n  回复: {response}")

            except KeyboardInterrupt:
                print("\n\n  已终止会话")
                break
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            print("\n  程序退出，清理资源。")


async def main():
    client = MCPClient()
    try:
        await client.connect_to_mcp_server()  # 连接mcp服务端
        await client.print_mcp_resources()  # 打印mcp服务端可用资源列表
        await client.chat_loop()  # 开始对话
    except Exception as e:
        print(f"\n⚠️ 程序出错: {str(e)}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
