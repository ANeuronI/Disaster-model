from groq import Groq
import streamlit as st
import re

GROQ_APIKEYS = ""
client = Groq(api_key=GROQ_APIKEYS)

# System prompt describing fuzzy logic rules for disaster assessment
fuzzy_logic_system_prompt = """
You are tasked with generating a detailed disaster assessment report based on a fuzzy logic system for disaster severity classification. The system considers the following key factors:

Severity Index:
- Ranges from 0 (no damage) to 3 (fully destroyed).
- Represents the overall intensity of damage across affected areas.

Damage Area Classifications:
- No damage (green)
- Light damage (yellow)
- Major damage (orange)
- Fully destroyed (red)
- Damage areas are further classified as small (0-30% of the total area), medium (30-60%), or large (60-100%).

Fuzzy Logic Rules for Disaster Classification:
1. **Minor Disaster**: If the severity index is low (0-1) and the fully destroyed area is small, classify the disaster as minor.
2. **Moderate Disaster**: If the severity index is moderate (1-2) and the fully destroyed area is medium, classify the disaster as moderate.
3. **Severe Disaster**: If the severity index is high (2-3) and the fully destroyed area is large, classify the disaster as severe. This requires urgent intervention.
4. **Localized but Critical**: If major damage is large but the fully destroyed area is small, classify the disaster as localized but critical, requiring focused attention.

Your task is to generate a detailed report that includes:
1. **Disaster Name and Type**: Name of the disaster and a relevant disaster type. For the **Disaster Type** field, categorize the disaster based on the integer part of the severity index, formatted as: `category ([Integer Index]) [Type of Disaster]`. Example: If the severity index is 2.3, categorize it as **category 2**.
2. **Severity Index and Impact**: Use the severity index and the damaged areas to provide an impact analysis.
3. **Damage Breakdown**: Provide a detailed breakdown of the areas affected by damage, including the percentages of no damage, light damage, major damage, and fully destroyed areas.
4. **Impact Analysis**: Analyze the severity and implications of the disaster using the fuzzy logic rules. Discuss the relationship between severity index and damage areas to justify the disaster classification (e.g., minor, moderate, severe).
5. **Recommended Actions**: Suggest next steps, such as evacuation, resource deployment, infrastructure assessment, or recovery efforts. Ensure that your recommendations align with the severity level and impacted areas.
6. **Timeline for Assistance**: Provide a structured timeline for response and recovery efforts, from immediate emergency actions to long-term rebuilding plans.

**Format for the Report**:

- **Disaster Report**: [Disaster Name]

- **Disaster Type**: category ([Integer Index]) [Type of Disaster]

- **Severity Index**: [Index] ([Low, Moderate, High])
  
**Summary**:
- Damage Assessment:
    - No Damage Area: [value] ([%] of total area)
    - Light Damage Area: [value] ([%] of total area)
    - Major Damage Area: [value] ([%] of total area)
    - Fully Destroyed Area: [value] ([%] of total area)
  
**Impact Analysis**:
- Provide a classification based on the severity index and fuzzy logic rules.
- Discuss the significance of the damage areas in relation to the total area.
- Compare the disaster’s damage levels against standard thresholds and fuzzy rules (e.g., a moderate severity index but high fully destroyed areas might elevate the disaster’s classification).
  
**Recommended Actions**:
- Specify immediate actions such as evacuations, medical aid, search and rescue operations, or securing affected areas.
- Highlight medium- and long-term strategies like resource allocation for reconstruction, monitoring efforts, and sustainability of recovery plans.

**Timeline for Assistance**:
- **Immediate Response (0-72 hours)**: Focus on critical life-saving measures, emergency shelters, medical aid, and provision of food and water.
- **Short-Term Recovery (72 hours - 1 week)**: Continue resource deployment for temporary repairs, shelter assistance, and debris removal.
- **Medium-Term Recovery (1-4 weeks)**: Implement structural repairs, infrastructure rebuilding, and restore essential services.
- **Long-Term Recovery (4 weeks - 3 months)**: Monitor ongoing recovery efforts, evaluate rebuilding progress, and support community resettlement and resource provision.

**Conclusion**:
Summarize the overall impact of the disaster and provide a final assessment. Stress the importance of timely response, long-term recovery strategies, and ensuring that affected communities receive the necessary support for full recovery.

Example Report:

**Disaster Report**: Hurricane Alex

**Disaster Type**: category (2) Hurricane

**Severity Index**: 2.5 (Moderate to High)

**Summary**:
- Damage Assessment:
    - No Damage Area: 1000 (20% of total area)
    - Light Damage Area: 2000 (40% of total area)
    - Major Damage Area: 1500 (30% of total area)
    - Fully Destroyed Area: 500 (10% of total area)

**Impact Analysis**:
- The severity index of 2.5 places the disaster in the moderate-to-high range.
- A fully destroyed area of 10%, combined with 30% major damage, indicates significant destruction.
- While the fully destroyed area is below the medium threshold, the extensive major damage suggests substantial impacts, classifying the disaster as moderate with serious localized destruction.

**Recommended Actions**:
1. Immediate evacuation of fully destroyed and severely damaged areas.
2. Deploy resources to areas with major damage for structural repairs.
3. Continue monitoring areas with light damage for recovery and prevent further damage.

**Timeline for Assistance**:
- **Immediate Response (0-72 hours)**: Search and rescue, medical aid, evacuation.
- **Short-Term Recovery (72 hours - 1 week)**: Temporary repairs, debris clearance.
- **Medium-Term Recovery (1-4 weeks)**: Infrastructure rebuilding and restoring essential services.
- **Long-Term Recovery (4 weeks - 3 months)**: Community resettlement, full restoration of services.

**Conclusion**:
- Hurricane Alex has caused significant destruction. Urgent action is necessary to mitigate further harm, ensure public safety, and facilitate the long-term recovery of affected communities.
"""



# Update history with fuzzy logic prompt
history = [
    {"role": "system", "content": fuzzy_logic_system_prompt},
]

# Function to handle chat and generate a disaster report
def Chat_handler_function(disaster_name, severity_index, damage_areas, total_building_area):
    # Prepare user input as context for the LLM
    user_input = f"""
    Disaster name: {disaster_name}
    Severity index: {severity_index}
    Total building area: {total_building_area}
    No damage area: {damage_areas['no_damage']}
    Light damage area: {damage_areas['light_damage']}
    Major damage area: {damage_areas['major_damage']}
    Fully destroyed area: {damage_areas['fully_destroyed']}
    
    Based on these inputs, generate a report explaining the type of disaster, its impact, and recommended actions.
    """

    # Append user input to the history
    history.append({"role": "user", "content": user_input})

    # Request LLM to generate a completion (disaster report)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Process the response in chunks and collect it
    new_message = {"role": "assistant", "content": ""}

    for chunk in completion:
        if chunk.choices[0].delta.content:
            new_message["content"] += chunk.choices[0].delta.content

    # Return the report generated by the LLM
    return new_message["content"]


def Report_visualiser(report):
    # Use regex to split the report into sections based on headings
    sections = re.split(r'(?=\*\*[^:]+:\*\*)', report)

    # Render each section
    for section in sections:
        if section.strip():  # Check if the section is not empty
            # Find the title and content
            title_match = re.match(r'\*\*([^:]+):\*\*', section)
            if title_match:
                title = title_match.group(1).strip()
                # content = section[title_match.end():].strip()
                
                # Render the title and content using Markdown
                st.markdown(f"<h4 style='font-size:20px;'>{title}</h4>", unsafe_allow_html=True)  # Use Markdown heading level 4
                # st.write(content)
            else:
                st.write(section)
      
        





