# tools_spec.py
# 20-tool schema for AI TMR Assistant
# - 실제 TMR 행동 패턴 기반 설계
# - LLM-in-tool 금지: 모든 tool은 DB/룰/템플릿/검색만 사용
#
# 전제 조건:
# - TMR은 전화 전 고객 기본 정보만 인지 (연령, 성별, 가입 상품)
# - 모든 추가 정보는 통화 중 고객이 말하는 것 기반
# - 본인 계약만 조회 가능 (배우자/자녀/제3자 조회 불가)
# - 콜/문의/납입/청구 이력 조회 불가 (개인정보)
# - 타사 비교 불가 (한국 공시 데이터 제한)
# - 회사: 당사 (치아보험/간편보험 주력)
#
# Function 기반 분류:
# F1: Information Retrieval (10 tools) - 상품/특약/인수 정보
# F2: Judgment/Calculation (4 tools)  - 판단/계산
# F3: Recommendation (4 tools)        - 추천
# F4: Compliance (2 tools)            - 규제/개인정보

from __future__ import annotations

TOOLS = [
    # ═══════════════════════════════════════════════════════════════
    # F1: Information Retrieval (10 tools)
    # ═══════════════════════════════════════════════════════════════

    # F1-01) customer_policy_lookup
    {
        "type": "function",
        "function": {
            "name": "customer_policy_lookup",
            "description": "전화 중인 본인 고객의 보유 보험 조회. ⚠️ 본인만 가능. 배우자/자녀/제3자 조회 불가. 콜/문의/납입/청구 이력 불가.",
            "parameters": {
                "type": "object",
                "required": ["customer_id"],
                "properties": {
                    "customer_id": {"type": "string", "description": "전화 중인 본인 고객 ID"},
                    "include_rejection_history": {
                        "type": "boolean",
                        "description": "가입 거절 이력 포함 여부 (선택)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-02) product_lookup
    {
        "type": "function",
        "function": {
            "name": "product_lookup",
            "description": "고객이 말한 조건에 맞는 우리 회사 보험 상품 검색.",
            "parameters": {
                "type": "object",
                "required": ["customer_age", "customer_gender", "coverage_interests"],
                "properties": {
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                    "coverage_interests": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": [
                                "cancer", "diagnosis", "hospitalization", "surgery",
                                "outpatient", "death", "CI", "disability",
                                "dental_general", "dental_implant", "dental_prosthetic",
                                "long_term_care"
                            ]
                        },
                        "description": "고객이 말한 관심 보장 (long_term_care=간병)"
                    },
                    "health_condition": {
                        "type": "string",
                        "enum": ["healthy", "minor_condition", "chronic_disease"],
                        "description": "고객이 말한 건강 상태 (선택)"
                    },
                    "smoking_status": {
                        "type": "string",
                        "enum": ["non_smoker", "smoker", "ex_smoker"],
                        "description": "고객이 말한 흡연 여부 (선택)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-03) coverage_detail_lookup
    {
        "type": "function",
        "function": {
            "name": "coverage_detail_lookup",
            "description": "특정 상품의 보장 세부사항 조회. 고객이 '이거 암 보장 어떻게 돼요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "coverage_categories"],
                "properties": {
                    "product_id": {"type": "string"},
                    "coverage_categories": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "enum": [
                                "cancer", "diagnosis", "hospitalization", "surgery",
                                "outpatient", "death", "CI", "disability",
                                "dental_general", "dental_implant", "dental_prosthetic",
                                "long_term_care", "exclusions"
                            ]
                        }
                    },
                    "include_exclusions": {
                        "type": "boolean",
                        "description": "제외사항 포함 여부"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-04) rider_detail_lookup
    {
        "type": "function",
        "function": {
            "name": "rider_detail_lookup",
            "description": "특약 상세 조회. 고객이 '진단금 특약 뭐예요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "rider_names"],
                "properties": {
                    "product_id": {"type": "string"},
                    "rider_names": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "enum": [
                                "cancer_rider", "diagnosis_rider", "hospitalization_rider",
                                "surgery_rider", "outpatient_rider", "CI_rider",
                                "dental_cavity_rider", "dental_implant_rider", "dental_prosthetic_rider"
                            ]
                        }
                    },
                    "include_premium": {
                        "type": "boolean",
                        "description": "특약 보험료 포함 여부"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-05) underwriting_rules_lookup
    {
        "type": "function",
        "function": {
            "name": "underwriting_rules_lookup",
            "description": "건강조건별 인수 규칙 조회. 고객이 병력을 말했을 때 할증/삭감 규칙 확인.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "health_conditions"],
                "properties": {
                    "product_id": {"type": "string"},
                    "health_conditions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "required": ["condition_name", "status"],
                            "properties": {
                                "condition_name": {
                                    "type": "string",
                                    "enum": [
                                        "hypertension", "diabetes", "thyroid", "cancer", "liver",
                                        "kidney", "heart", "stroke", "mental", "other"
                                    ]
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["current", "past_cured", "past_treating", "family_history"]
                                },
                            },
                        }
                    },
                    "return_detail_level": {
                        "type": "string",
                        "enum": ["summary", "detailed"]
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-06) underwriting_policy_qa
    {
        "type": "function",
        "function": {
            "name": "underwriting_policy_qa",
            "description": "TMR이 헷갈리는 최신 인수 정책 Q&A. '갑상선 무투약 기준 뭐였지?' 같은 경우.",
            "parameters": {
                "type": "object",
                "required": ["query_topics"],
                "properties": {
                    "query_topics": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": [
                                "diabetes_threshold_2024", "thyroid_no_med_rule",
                                "cancer_5y_rule", "hypertension_medication_change",
                                "mental_health_new_policy", "covid_aftermath_rule",
                                "smoking_verification", "family_history_scope"
                            ]
                        }
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-07) waiting_period_lookup
    {
        "type": "function",
        "function": {
            "name": "waiting_period_lookup",
            "description": "대기기간/면책/감액 규정 조회. 고객이 '가입하면 바로 보장돼요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "coverage_type"],
                "properties": {
                    "product_id": {"type": "string"},
                    "coverage_type": {
                        "type": "string",
                        "enum": [
                            "cancer", "diagnosis", "hospitalization", "surgery",
                            "CI", "death", "general_disease"
                        ],
                        "description": "보장 유형"
                    },
                    "rule_type": {
                        "type": "string",
                        "enum": ["waiting_period", "reduction", "exclusion"],
                        "description": "규칙 유형 (선택, 기본 전체)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-08) health_declaration_guide
    {
        "type": "function",
        "function": {
            "name": "health_declaration_guide",
            "description": "건강고지 가이드 조회. 고객이 어떤 건강 정보를 고지해야 하는지 안내할 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "condition_category"],
                "properties": {
                    "product_id": {"type": "string"},
                    "condition_category": {
                        "type": "string",
                        "enum": [
                            "hypertension", "diabetes", "thyroid", "cancer",
                            "liver", "kidney", "heart", "mental",
                            "surgery_history", "hospitalization_history", "medication"
                        ],
                        "description": "고객이 말한 질환/상태 카테고리"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-09) product_comparison
    {
        "type": "function",
        "function": {
            "name": "product_comparison",
            "description": "우리 회사 상품 간 비교. ⚠️ 타사 비교 불가. 고객이 '이거랑 저거 차이 뭐예요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_ids", "comparison_aspects"],
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 3,
                        "items": {"type": "string"},
                        "description": "비교할 우리 회사 상품 ID"
                    },
                    "comparison_aspects": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "enum": [
                                "premium", "coverage_scope", "coverage_amount",
                                "exclusions", "underwriting_requirements"
                            ]
                        }
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F1-10) product_change_calculator
    {
        "type": "function",
        "function": {
            "name": "product_change_calculator",
            "description": "갈아타기 시 손익 계산. 기존 상품 해약 vs 새 상품 가입 비교.",
            "parameters": {
                "type": "object",
                "required": ["current_policy_id", "new_product_id", "customer_age"],
                "properties": {
                    "current_policy_id": {"type": "string", "description": "현재 보유 상품 ID"},
                    "new_product_id": {"type": "string", "description": "갈아탈 상품 ID"},
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                },
                "additionalProperties": False,
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════
    # F2: Judgment/Calculation (4 tools)
    # ═══════════════════════════════════════════════════════════════

    # F2-01) underwriting_eligibility_checker
    {
        "type": "function",
        "function": {
            "name": "underwriting_eligibility_checker",
            "description": "가입 가능성 판단. 고객이 병력/흡연 등을 말했을 때 '이 상품 가입 돼요?' 확인.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "disclosed_conditions", "customer_age"],
                "properties": {
                    "product_id": {"type": "string"},
                    "disclosed_conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["condition_name", "status", "severity"],
                            "properties": {
                                "condition_name": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["current", "past_cured", "past_treating"]
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["mild", "moderate", "severe"]
                                },
                            },
                        },
                        "maxItems": 5,
                        "description": "고객이 고지한 건강 상태"
                    },
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "smoking_status": {
                        "type": "string",
                        "enum": ["non_smoker", "smoker", "ex_smoker"],
                        "description": "고객이 말한 흡연 여부 (선택)"
                    },
                    "return_alternatives": {
                        "type": "boolean",
                        "description": "불가 시 대안 상품 제시 여부"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F2-02) premium_calculator
    {
        "type": "function",
        "function": {
            "name": "premium_calculator",
            "description": "보험료 계산. 고객이 '보험료 얼마예요?' 물어볼 때 빠르게 계산.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "customer_age", "customer_gender", "payment_period", "coverage_amount"],
                "properties": {
                    "product_id": {"type": "string"},
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                    "payment_period": {
                        "type": "string",
                        "enum": ["monthly", "quarterly", "semi_annual", "annual"]
                    },
                    "coverage_amount": {
                        "type": "integer",
                        "minimum": 10000000,
                        "maximum": 500000000,
                        "description": "보장 금액 (원)"
                    },
                    "health_surcharge": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "description": "건강 할증률 (0=없음)"
                    },
                    "smoking_surcharge": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.5,
                        "description": "흡연 할증률 (0=없음)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F2-03) smoking_impact_calculator
    {
        "type": "function",
        "function": {
            "name": "smoking_impact_calculator",
            "description": "흡연 보험료 차이 계산. 고객이 '담배 피우면 얼마나 비싸져요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "customer_age", "customer_gender", "smoking_status"],
                "properties": {
                    "product_id": {"type": "string"},
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                    "smoking_status": {
                        "type": "string",
                        "enum": ["non_smoker", "smoker", "ex_smoker_1y", "ex_smoker_3y", "ex_smoker_5y"],
                        "description": "고객이 말한 흡연 상태"
                    },
                    "coverage_amount": {
                        "type": "integer",
                        "minimum": 10000000,
                        "maximum": 500000000
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F2-04) special_condition_lookup
    {
        "type": "function",
        "function": {
            "name": "special_condition_lookup",
            "description": "조건부 인수 시 특별 조건 상세 조회. '할증 몇% 붙어요?' '어떤 보장 제한 걸려요?' 같은 경우.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "condition_type"],
                "properties": {
                    "product_id": {"type": "string"},
                    "condition_type": {
                        "type": "string",
                        "enum": [
                            "surcharge_detail", "coverage_exclusion",
                            "coverage_reduction", "waiting_extension",
                            "conditional_approval_terms"
                        ],
                        "description": "조건부 인수 유형"
                    },
                    "health_condition": {
                        "type": "string",
                        "description": "해당 건강 상태 (예: hypertension)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════
    # F3: Recommendation (4 tools)
    # ═══════════════════════════════════════════════════════════════

    # F3-01) product_recommender
    {
        "type": "function",
        "function": {
            "name": "product_recommender",
            "description": "고객 니즈/예산 기반 상품 추천. 고객이 예산이랑 니즈를 말했을 때.",
            "parameters": {
                "type": "object",
                "required": ["customer_needs", "budget_monthly"],
                "properties": {
                    "customer_needs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                        "description": "고객이 말한 니즈"
                    },
                    "budget_monthly": {
                        "type": "integer",
                        "minimum": 10000,
                        "maximum": 500000,
                        "description": "고객이 말한 월 예산 (원)"
                    },
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                    "health_profile": {
                        "type": "string",
                        "enum": ["healthy", "minor_condition", "chronic_disease"]
                    },
                    "smoking_status": {
                        "type": "string",
                        "enum": ["non_smoker", "smoker", "ex_smoker"]
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F3-02) rider_recommendation
    {
        "type": "function",
        "function": {
            "name": "rider_recommendation",
            "description": "특약 추가 추천. 고객이 '뭐 더 추가하면 좋아요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["product_id", "customer_needs"],
                "properties": {
                    "product_id": {"type": "string"},
                    "customer_needs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                        "description": "고객이 말한 추가 니즈"
                    },
                    "current_riders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                        "description": "현재 가입된 특약"
                    },
                    "budget_additional": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100000,
                        "description": "고객이 말한 추가 예산 (월, 원)"
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F3-03) policy_upsell_checker
    {
        "type": "function",
        "function": {
            "name": "policy_upsell_checker",
            "description": "보유 상품 기반 업셀/크로스셀/갈아타기 기회 탐색.",
            "parameters": {
                "type": "object",
                "required": ["customer_id", "current_policies"],
                "properties": {
                    "customer_id": {"type": "string"},
                    "current_policies": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "required": ["policy_id", "product_type"],
                            "properties": {
                                "policy_id": {"type": "string"},
                                "product_type": {
                                    "type": "string",
                                    "enum": ["health", "cancer", "CI", "death", "annuity"]
                                },
                                "subscription_date": {"type": "string", "description": "가입일 (YYYY-MM-DD)"},
                            },
                        },
                        "description": "customer_policy_lookup 결과"
                    },
                    "check_scope": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["upsell", "cross_sell", "replacement", "rider_addition"]
                        }
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F3-04) budget_optimizer
    {
        "type": "function",
        "function": {
            "name": "budget_optimizer",
            "description": "예산 내 최적 상품/특약 조합. 고객이 '이 돈으로 암이랑 입원 다 돼요?' 물어볼 때.",
            "parameters": {
                "type": "object",
                "required": ["customer_age", "customer_gender", "budget_monthly", "priority_coverages"],
                "properties": {
                    "customer_age": {"type": "integer", "minimum": 18, "maximum": 100},
                    "customer_gender": {"type": "string", "enum": ["male", "female"]},
                    "budget_monthly": {
                        "type": "integer",
                        "minimum": 10000,
                        "maximum": 500000,
                        "description": "고객이 말한 월 예산 (원)"
                    },
                    "priority_coverages": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": ["cancer", "diagnosis", "hospitalization", "surgery", "death", "CI", "long_term_care"]
                        },
                        "description": "고객이 말한 우선순위 보장"
                    },
                    "health_profile": {
                        "type": "string",
                        "enum": ["healthy", "minor_condition", "chronic_disease"]
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # ═══════════════════════════════════════════════════════════════
    # F4: Compliance (2 tools)
    # ═══════════════════════════════════════════════════════════════

    # F4-01) compliance_checker
    {
        "type": "function",
        "function": {
            "name": "compliance_checker",
            "description": "규정 위반 여부 검증. '이렇게 말해도 되나?' 같은 경우.",
            "parameters": {
                "type": "object",
                "required": ["tmr_statement_or_action", "check_categories"],
                "properties": {
                    "tmr_statement_or_action": {
                        "type": "string",
                        "description": "TMR이 하려는 말이나 행동"
                    },
                    "check_categories": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "enum": [
                                "misleading_info", "exaggeration", "guarantee_claim",
                                "inappropriate_pressure", "data_privacy", "disclosure_requirement"
                            ]
                        }
                    },
                },
                "additionalProperties": False,
            },
        },
    },

    # F4-02) privacy_violation_detector
    {
        "type": "function",
        "function": {
            "name": "privacy_violation_detector",
            "description": "개인정보 위반 검증. TMR이 조회 불가 데이터를 요청할 때 차단.",
            "parameters": {
                "type": "object",
                "required": ["requested_data_types", "data_subject"],
                "properties": {
                    "requested_data_types": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {
                            "type": "string",
                            "enum": [
                                "policy_owned", "rejection_history",
                                "call_history", "inquiry_history",
                                "payment_history", "claim_history",
                                "medical_record", "credit_score"
                            ]
                        },
                        "description": "⚠️ 허용: policy_owned, rejection_history만. 나머지 전부 위반"
                    },
                    "data_subject": {
                        "type": "string",
                        "enum": ["self", "spouse", "child", "third_party"],
                        "description": "⚠️ self만 허용. 나머지 전부 위반"
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["sales", "underwriting", "service"]
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


if __name__ == "__main__":
    print(f"Loaded {len(TOOLS)} tools:")
    for i, t in enumerate(TOOLS, 1):
        print(f"  {i:2d}. {t['function']['name']}")
