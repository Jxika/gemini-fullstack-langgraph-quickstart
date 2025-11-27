import os
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field
from langchain.tools import tool
import requests
import json
# ============ 全球临床试验查询 Tool（仅方法与入参，无实现） ============

class GlobalClinicalTrialsQueryInput(BaseModel):
    target: Optional[str] = Field(
        default=None,
        description="靶点查询：试验药靶点（例如 'PD-1'）。为空表示不按靶点筛选。",
    )
    drug: Optional[str] = Field(
        default=None,
        description="药物查询：所有试验药。为空表示不按药物筛选。",
    )
    company: Optional[str] = Field(
        default=None,
        description="企业查询：申办者/合作者。为空表示不按企业筛选。",
    )
    disease: Optional[str] = Field(
        default=None,
        description="疾病查询：适应症。为空表示不按疾病筛选。",
    )



class GlobalClinicalTrialsResultItem(BaseModel):
    登记号: Optional[str] = Field(None, description="临床登记号")
    试验药通用名: Optional[str] = Field(None, description="试验药通用名")
    试验药靶点: Optional[str] = Field(None, description="试验药靶点")
    药品类型: Optional[str] = Field(None, description="药品类型")
    标准适应症: Optional[str] = Field(None, description="标准适应症")
    申办者: Optional[str] = Field(None, description="申办者")
    合作者: Optional[str] = Field(None, description="合作者")
    首次公示日期: Optional[str] = Field(None, description="首次公示日期（YYYY-MM-DD）")
    试验分期: Optional[str] = Field(None, description="试验分期")
    试验状态: Optional[str] = Field(None, description="试验状态")
    结果评价: Optional[str] = Field(None, description="结果评价")
    DOI号: Optional[str] = Field(None, description="DOI号（在详情页的文献摘要里）")


class GlobalClinicalTrialsStats(BaseModel):
    总条目数: int = Field(..., description="统计总条目数")
    分期条目统计: dict = Field(
        default_factory=dict,
        description="按不同试验分期统计的条目总数，如 {'I':12,'II':34}"
    )
    状态条目统计: dict = Field(
        default_factory=dict,
        description="按不同试验状态统计的条目总数，如 {'Recruiting':20,'Completed':15}"
    )


class GlobalClinicalTrialsOutput(BaseModel):
    列表: List[GlobalClinicalTrialsResultItem] = Field(
        default_factory=list,
        description="符合筛选条件的试验列表条目。",
    )
    统计: GlobalClinicalTrialsStats = Field(
        ..., description="总体统计值与分组统计结果。"
    )


@tool("search_global_clinical_trials",return_direct=False,args_schema=GlobalClinicalTrialsQueryInput,description="全球临床试验")
def search_global_clinical_trials(
    target: Optional[str] = None,
    drug: Optional[str] = None,
    company: Optional[str] = None,
    disease: Optional[str] = None,
) -> GlobalClinicalTrialsOutput:
    """
    全球临床试验；
    入参包含四个维度（可任选并组合）：
    - 靶点：试验药靶点 
    - 药物：所有试验药
    - 企业：申办者/合作者
    - 疾病：适应症

    出参包含：
    - 列表：[{临床登记号, 试验药通用名, 试验药靶点, 药品类型, 标准适应症, 申办者, 合作者, 首次公示日期, 试验分期, 试验状态, 结果评价, DOI号}]
    - 统计：总条目数、不同试验分期条目总数、不同试验状态条目总数
    """
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
    }
    url="http://172.16.66.26:5000/api/Clinical/GetTableListForAI"
    params={"Target": target, "Drug": drug, "Enterprise": company, "Disease": disease,"pageSize":20}
    resp=requests.post(url, json=params,headers=headers)
    data=json.loads(resp.text)

    print(resp.json)

  

    raise NotImplementedError("search_global_clinical_trials: 仅定义方法与入参/出参，占位未实现")


    # ============ 临床试验结果查询 Tool（仅方法与入参，无实现） ============

class ClinicalTrialResultsQueryInput(BaseModel):
        registration_id: Optional[str] = Field(
            default=None, description="临床登记号（例如 NCT 番号或本地登记号），为空表示不限。"
        )
        target: Optional[str] = Field(
            default=None, description="靶点，为空表示不限。"
        )
        drug: Optional[str] = Field(
            default=None, description="药物英文/通用名，为空表示不限。"
        )
        company: Optional[str] = Field(
            default=None, description="企业（申办者/合作者），为空表示不限。"
        )
        disease: Optional[str] = Field(
            default=None, description="疾病（标准适应症/包含子适应症），为空表示不限。"
        )

class ClinicalTrialPublicationResultItem(BaseModel):
        文献标题: Optional[str] = Field(None, description="文献标题")
        临床登记号: Optional[str] = Field(None, description="临床登记号")
        药物: Optional[str] = Field(None, description="药物英文/通用名")
        药物中文: Optional[str] = Field(None, description="药物中文名称")
        终点指标: Optional[str] = Field(None, description="终点指标")
        结果评价: Optional[str] = Field(None, description="结果评价")
        试验分期: Optional[str] = Field(None, description="试验分期")
        申办者: Optional[str] = Field(None, description="申办者")
        合作者: Optional[str] = Field(None, description="合作者")
        药物靶点: Optional[str] = Field(None, description="药物靶点")
        药品类型: Optional[str] = Field(None, description="药品类型")
        生物标志物: Optional[str] = Field(None, description="生物标志物")
        标准适应症: Optional[str] = Field(None, description="标准适应症")
        文献详情链接: Optional[str] = Field(None, description="文献详情链接（URL）")

@tool("search_clinical_trial_results",return_direct=False,args_schema=ClinicalTrialResultsQueryInput,description="查询近年发表的临床试验结果")
def search_clinical_trial_results(
        target: Optional[str] = None,
        drug: Optional[str] = None,
        company: Optional[str] = None,
        disease: Optional[str] = None,
    ) -> List[ClinicalTrialPublicationResultItem]:
        """
        入参：
        - 靶点 target
        - 药物 drug
        - 企业 company（申办者/合作者）
        - 疾病 disease（适应症）

        出参（列表，每项包含）：
        { 文献标题、临床登记号、药物、药物中文、终点指标、结果评价、试验分期、申办者、合作者、药物靶点、药品类型、生物标志物、标准适应症、文献详情链接 }

        说明：仅定义方法与入参/出参结构，不含具体实现。
        """
        url="http://172.16.66.26:5000/api/ClinicalOutcomes/GetTableListForAI"
        params={"Target": target, "Drug": drug, "Enterprise": company, "Disease": disease,"pageSize":50}
        resp=requests.get(url, params=params)
        print(resp.json)

        raise NotImplementedError("search_clinical_trial_results: 仅定义方法与入参/出参，占位未实现")




# ============ 全球药物研发（临床阶段项目）查询 Tool（仅方法与入参，无实现） ============

class GlobalDrugRNDQueryInput(BaseModel):
    target: Optional[str] = Field(default=None, description="靶点，为空表示不限。")
    drug: Optional[str] = Field(default=None, description="药物通用名/化学名，为空表示不限。")
    company: Optional[str] = Field(default=None, description="企业（原研或合作），为空表示不限。")
    disease: Optional[str] = Field(default=None, description="疾病（全球或国内适应症），为空表示不限。")
    page: int = Field(default=1, description="分页页码，从 1 开始。")
    page_size: int = Field(default=50, description="分页大小，默认 50，建议不超过 200。")


class GlobalDrugRNDResultItem(BaseModel):
    项目名称: Optional[str] = Field(None, description="项目名称")
    药品通用名: Optional[str] = Field(None, description="药品通用名")
    项目创新程度: Optional[str] = Field(None, description="项目创新程度")
    原研企业: Optional[str] = Field(None, description="原研企业")
    合作企业: Optional[str] = Field(None, description="合作企业")
    药品类型: Optional[str] = Field(None, description="药品类型")
    靶点: Optional[str] = Field(None, description="靶点")
    作用机制: Optional[str] = Field(None, description="作用机制")
    项目最高阶段: Optional[str] = Field(None, description="项目最高研发阶段")
    项目研发状态: Optional[str] = Field(None, description="项目研发状态")
    全球研发适应症: Optional[str] = Field(None, description="全球研发适应症")
    中国内地研发适应症: Optional[str] = Field(None, description="中国内地研发适应症")
    境外研发适应症: Optional[str] = Field(None, description="境外（中国外）研发适应症")
    临床登记号: Optional[str] = Field(None, description="相关临床试验登记号")


@tool("search_global_drug_rnd",return_direct=False,args_schema=GlobalDrugRNDQueryInput,description="查询全球药物研发")
def search_global_drug_rnd(
    target: Optional[str] = None,
    drug: Optional[str] = None,
    company: Optional[str] = None,
    disease: Optional[str] = None
) -> List[GlobalDrugRNDResultItem]:
    """
    入参：靶点 target，药物 drug，企业 company，疾病 disease（适应症)。

    出参：列表，每项字段包括：
    {项目名称、药品通用名、项目创新程度、原研企业、合作企业、药品类型、靶点、作用机制、项目最高阶段、项目研发状态、全球研发适应症、中国内地研发适应症、境外研发适应症、临床登记号}

    说明：仅定义方法与入参/出参结构，不含具体实现。
    """
    url="http://172.16.66.26:5000/api/GlobalNewDrug/GetTableListForAI"
    params={"Target": target, "Drug": drug, "Enterprise": company, "Disease": disease, "pageSize": 50}
    resp=requests.get(url, params=params)

    raise NotImplementedError("search_global_drug_rnd: 仅定义方法与入参/出参，占位未实现")



if __name__ == "__main__":
    # 测试临床试验查询

    search_global_clinical_trials.invoke({"target": "PD-1"})

    # 测试临床试验结果查询
    # results = search_clinical_trial_results(target="PD-1", drug="Nivolumab")
    # print(results)

    # # 测试全球药物研发查询
    # rnd_results = search_global_drug_rnd(target="PD-1", drug="Nivolumab")
    # print(rnd_results)