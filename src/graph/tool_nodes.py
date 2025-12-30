# src/graph/tool_nodes.py
from typing import Dict, Any
from src.graph.state import AgentState
from src.utils.logger import logger
from src.core.llm_client import llm_client


class ToolDecisionNode:
    """决定是否需要以及如何使用工具"""

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        user_input = state["user_input"]
        logger.info(f"工具决策节点处理: {user_input}")

        # 使用LLM判断是否需要工具
        tool_decision = self._decide_tool_use(user_input)

        if tool_decision["need_tool"]:
            return {
                "should_use_tool": True,
                "tool_name": tool_decision.get("tool_name"),
                "tool_args": tool_decision.get("arguments", {}),
                "reasoning": tool_decision.get("reasoning", ""),
                "current_step": "tool_decision"
            }
        else:
            return {
                "should_use_tool": False,
                "current_step": "tool_decision"
            }

    def _decide_tool_use(self, query: str) -> Dict[str, Any]:
        """调用LLM分析是否需要工具"""
        prompt = f"""
        分析用户问题，判断是否需要调用外部工具/API。

        可用工具列表：
        1. 计算器 (calculator): 执行数学计算，如 "2+3*4"
        2. 天气查询 (get_weather): 查询城市天气，需要参数 city
        3. 单位转换 (unit_converter): 转换单位，如 "10公里换算成英里"

        如果不需要工具，返回：{{"need_tool": false}}
        如果需要工具，返回JSON：{{"need_tool": true, "tool_name": "工具名", "arguments": {{"参数": "值"}}}}

        问题：{query}
        只返回JSON，不要其他文本。
        """

        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            import json
            decision = json.loads(response["content"].strip())
            return decision
        except Exception as e:
            logger.error(f"工具决策失败: {e}")
            return {"need_tool": False}


class ToolExecutionNode:
    """执行具体的工具调用"""

    def __init__(self):
        # 初始化工具实现
        self.tools = {
            "calculator": self._calculate,
            "get_weather": self._get_weather,
            "unit_converter": self._convert_units
        }

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        tool_name = state.get("tool_name")
        tool_args = state.get("tool_args", {})

        if not tool_name or tool_name not in self.tools:
            return {
                "tool_result": {"error": f"未知工具: {tool_name}"},
                "current_step": "tool_execution"
            }

        logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")

        # 执行工具
        try:
            result = self.tools[tool_name](**tool_args)
            return {
                "tool_result": result,
                "tool_used": tool_name,
                "current_step": "tool_execution"
            }
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return {
                "tool_result": {"error": str(e)},
                "current_step": "tool_execution"
            }

    def _calculate(self, expression: str = ""):
        """简单的计算器实现"""
        try:
            # 安全计算 - 使用ast.literal_eval替代eval
            import ast
            # 清理表达式
            cleaned_expr = ''.join(c for c in expression if c.isdigit() or c in '+-*/.() ')
            result = ast.literal_eval(cleaned_expr)
            return {"result": result, "expression": expression}
        except:
            return {"error": "无法计算表达式", "expression": expression}

    def _get_weather(self, city: str = "北京"):
        """模拟天气查询"""
        # 实际项目中这里会调用天气API
        weather_data = {
            "北京": {"temp": "22°C", "condition": "晴", "humidity": "40%"},
            "上海": {"temp": "25°C", "condition": "多云", "humidity": "65%"},
            "深圳": {"temp": "28°C", "condition": "阵雨", "humidity": "80%"}
        }

        if city in weather_data:
            return {"city": city, **weather_data[city]}
        else:
            return {"city": city, "temp": "24°C", "condition": "未知", "note": "模拟数据"}

    def _convert_units(self, value: float = 1.0, from_unit: str = "km", to_unit: str = "mile"):
        """单位转换"""
        conversions = {
            ("km", "mile"): 0.621371,
            ("mile", "km"): 1.60934,
            ("kg", "lb"): 2.20462,
            ("lb", "kg"): 0.453592,
            ("c", "f"): lambda c: c * 9 / 5 + 32,
            ("f", "c"): lambda f: (f - 32) * 5 / 9
        }

        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            conv = conversions[key]
            if callable(conv):
                result = conv(value)
            else:
                result = value * conv

            return {
                "result": round(result, 4),
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
        else:
            return {"error": f"不支持的单位转换: {from_unit} → {to_unit}"}