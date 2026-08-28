"""
Rule Creator: the offline pipeline that mines canonical SQL-correction
rules from BIRD *training* data.

See README.md in this directory for the four-step methodology (error
mining -> database-agnostic filtering -> clustering -> rule synthesis) and
dextersql.rule_creator.pipeline for the orchestrator/CLI entrypoint.

This package never reads dev/test data -- see config.RuleCreatorConfig.
"""
