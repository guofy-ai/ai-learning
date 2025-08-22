import random
from datetime import datetime
from typing import Annotated
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# 本地调试方式
# 启动命令：mcp dev server.py
# 在弹出的网页中可以对resources、tools、prompts进行调试

# 初始化 FastMCP 服务器
mcp = FastMCP("我的测试Demo")


# ===== 资源定义 =====

@mcp.resource(uri="city-tourist-doc://{city}/city-tourist.md", description="景点介绍")
def city_tourist(city: str) -> str:
    city = unquote(city, encoding='utf-8')
    return _read_file_content(f"./resources/{city}景点推荐.md")  # 根据输入城市，读取对应文档


# ===== 工具定义 =====

@mcp.tool(description="天气查询工具")
def get_current_weather(
        location: Annotated[str, Field(description="城市名称")]
) -> str:
    weather_conditions = ["晴天", "多云", "雨天"]
    random_weather = random.choice(weather_conditions)
    return f"{location}今天是{random_weather}。"


@mcp.tool(description="查询当前时间的工具")
def get_current_time():
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前时间：{formatted_time}。"


# ===== 提示模板定义 =====

@mcp.prompt(description="北京景点推荐")
def top_beijing_tourist_spots(city_info: str) -> str:
    return f"你是旅游博主，非常了解北京文化，按以下内容进行推荐\n\n{city_info}"


@mcp.prompt(description="北京美食推荐")
def top_beijing_fine_food(city_info: str) -> str:
    return f"你是美食博主，非常了解北京饭馆，按以下内容进行推荐\n\n{city_info}"


# ===== 辅助函数 =====

def _read_file_content(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取文件 {file_path} 失败: {str(e)}"


# ===== 主程序入口 =====

if __name__ == "__main__":
    print("以标准 I/O 方式运行 MCP 服务器")
    mcp.run(transport='stdio')
