"""
任务分类器 - 专门负责判断用户请求的类型

职责：
1. 接收用户输入，分析其意图
2. 根据预定义的分类规则，将任务归类为：
   - appointment（预约任务）
   - query（查询任务）  
   - pay（支付任务）
   - statistics（统计任务）
   - other（其他任务）
3. 提供清晰的分类结果和置信度
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any


class TaskClassifier:
    """任务分类器 - 使用LLM进行智能任务分类"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initialize_prompt()
        self.chain = self.prompt | self.llm
    
    def _initialize_prompt(self):
        """初始化分类提示词模板"""
        self.prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "给门店服务请求分类。只输出一个英文标签。\n"
                "query：问门店地址、营业时间、如何到店、服务介绍、价格或工作人员信息。\n"
                "appointment：请求安排服务，或变更本次服务时间、时长、人员。\n"
                "pay：付款、结账，或预约机器人转来的已确认服务。\n"
                "statistics：工作人员上报服务已完成，或请求业务统计。\n"
                "other：与门店业务无关，例如写程序或闲聊。\n"
                "门店位置怎么找？ → query\n"
                "请安排今天下午的推拿 → appointment\n"
                "用户输入：{task}\n"
                "标签："
            )
        )
    
    async def classify_task(self, task: str) -> str:
        """
        分类任务
        
        Args:
            task: 用户输入的任务内容
            
        Returns:
            str: 分类结果 ('appointment', 'query', 'pay', 'statistics', 'other')
        """
        try:
            category_msg = await self.chain.ainvoke({"task": task})
            category = category_msg.content.strip().lower()
            
            # 验证分类结果是否有效
            valid_categories = {'appointment', 'query', 'pay', 'statistics', 'other'}
            if category not in valid_categories:
                return 'other'  # 默认归类为其他
                
            return category
            
        except Exception as e:
            print(f"任务分类失败: {str(e)}")
            return 'other'  # 发生错误时默认归类为其他
    
    def get_category_description(self, category: str) -> str:
        """获取分类类别的描述信息"""
        descriptions = {
            'appointment': '预约任务 - 用户或工作人员的预约相关请求',
            'query': '查询任务 - 用户咨询服务信息、价格、工作人员等',
            'pay': '支付任务 - 完成预约后的支付相关事务',
            'statistics': '统计任务 - 工作人员上报工作完成状态',
            'other': '其他任务 - 与按摩服务无关的请求'
        }
        return descriptions.get(category, '未知任务类型')
