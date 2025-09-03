def generate_ontology_description(ontology: dict, current_step: str) -> str:
    """Generates a markdown string describing the ontology."""
    desc = ""
    if current_step=="answer":
        return desc
    nodes = ontology.get(current_step, {})
    for node_name, node_info in nodes.items():
        desc += f"- **{node_name}**: {node_info['description']}\n"
        if "attributes" in node_info:
            for attr_name, attr_info in node_info["attributes"].items():
              if "options" in attr_info:
                options = ", ".join([f"`{opt}`" for opt in attr_info["options"]])
                desc += f"  - `{attr_name}` ({attr_info['type']}): {attr_info['description']} Options: [{options}].\n"
              else:
                desc += f"  - `{attr_name}` ({attr_info['type']}): {attr_info['description']}.\n"
    return desc

# New, more sophisticated prompt template
COMBINED_ANNOTATION_PROMPT_TEMPLATE = """
You are an expert human annotator analyzing AI agent behavior. Your task is to annotate a step in an agent's reasoning trace using balanced, practical judgment criteria.

**1. Ontology Definition**
{ontology_str}

**2. Trace History (Previous Steps)**
Here are the steps that occurred *before* the current step.
```json
{previous_steps_json}
```

**3. Current Step to Annotate**
This is the step you must analyze and annotate. Its index is `{current_step_index}`.
```json
{current_step_json}
```
{search_count_info}

**4. Balanced Analysis Process**

**Step 1: Understand the Step's Purpose**
What is the agent doing in this step?
- Is it assessing what it knows/doesn't know?
- Is it making a plan for action?
- Is it synthesizing information from search results?
- Is it critiquing and correcting previous reasoning?
- Is it formulating a search query?

**Step 2: Map to Ontology Type**
Based on the step's purpose, select the most appropriate type from the ontology above.
The ontology type should match what the agent is primarily doing in this step.

**Step 3: Evaluate Attributes (Balanced Approach)**

**For `information_clarity`:**
- **Be Practical**: Does this contain relevant information for the task?
- **Accept Partial Information**: If relevant information is present, mark as "Clear"
- **Be Reasonable**: Don't require perfect or complete information
- **Only Mark Ambiguous**: If there are truly conflicting or confusing answers
- **Question Relevance**: Focus on whether the information helps answer the question

**For `premise_grounding`:**
- **Include Planning with Facts**: Planning statements that mention specific entities, locations, or concepts
- **Accept Indirect Evidence**: If the statement relates to the question and has some evidence support
- **Question Relevance**: If the statement contains facts relevant to the question, it's grounded
- **Context Matters**: Consider what the agent is trying to accomplish
- **Examples**:
  - "I need to find out where Leo Bennett died" → **Directly Grounded** (mentions specific person and concept)
  - "I need to think about this" → **Not Grounded** (no specific facts)
  - "Based on the search, Leo Bennett died in Thames Ditton" → **Directly Grounded** (explicit factual claim)

**For other attributes:**
- **Use Balanced Judgment**: Be reasonable and pragmatic
- **Consider Context**: Think about the practical context of the agent's task
- **Be Consistent**: Apply similar standards across similar situations

**Step 4: Determine Dependency**
Which single prior step (by step_index) most directly led to or enabled this current step?
- For the first step: use null
- Otherwise: identify the step that provided the information or context that triggered this action

**Balanced Annotation Guidelines:**
- **Be Practical**: Focus on what the agent is actually doing, not theoretical perfection
- **Be Reasonable**: Use balanced judgment rather than overly strict or overly lenient standards
- **Consider Context**: Think about the practical context of the agent's task
- **Question Relevance**: Focus on whether the step helps answer the original question
- **Evidence Tolerance**: Accept indirect or partial evidence support when reasonable

**5. Output Format**
Your response **MUST** be a single, valid JSON object with two keys: `annotation` and `trace_dependency`.

```json
{{
  "annotation": {{
    "type": "<exact_type_name_from_ontology>",
    "justification": "Brief explanation using balanced reasoning, referencing specific evidence from the step content.",
    "attributes": {{
      "<attribute_name>": "<value_from_ontology_options>",
      "<another_attribute>": "<value_from_ontology_options>"
    }}
  }},
  "trace_dependency": {{
    "dependent_on": <index_of_prior_step_or_null>
  }}
}}
```
""" 