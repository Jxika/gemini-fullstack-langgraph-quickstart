# swagger_tools_config.py

swagger_tools = [
    {
        "name": "get_clinical_results",
        "description": (
            "Use this tool when you need structured clinical research or trial results data. "
            "It queries the global clinical trial results dataset. "
            "API endpoint: /api/Clinical/GetTableList"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Keywords or condition name to filter clinical trials (e.g., 'breast cancer', 'COVID-19 vaccine')."
                },
                "country": {
                    "type": "string",
                    "description": "Optional country filter for trials."
                },
                "year": {
                    "type": "integer",
                    "description": "Optional year filter for results."
                },
            },
            "required": ["keywords"]
        },
    },
    {
        "name": "get_global_drug_patents",
        "description": (
            "Use this tool when you need drug patent data or legal status for pharmaceuticals. "
            "It queries the global drug patent dataset. "
            "API endpoint: /api/GlobalDrugPatents/GetTableList"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The name of the drug to look up patents for."
                },
                "country": {
                    "type": "string",
                    "description": "Optional patent jurisdiction, e.g., 'US', 'CN', 'EU'."
                },
                "status": {
                    "type": "string",
                    "description": "Patent status filter, e.g., 'Active', 'Expired'."
                },
            },
            "required": ["drug_name"]
        },
    }
]
