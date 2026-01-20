#创建游戏详情的用户界面：
import streamlit as st
import autogen
from autogen.agentchat import GroupChat, GroupChatManager

if 'output' not in st.session_state:
    st.session_state.output = {
        'story': '', 'gameplay': '',
        'visuals': '', 'tech': ''
    }

#创建游戏详情的用户界面:
st.title("AI Game Design Agent Team")
col1, col2 = st.columns(2)

with col1:
    background_vibe = st.text_input("Background Vibe")
    game_type = st.selectbox("Game Type", ["RPG", "Action", "Adventure"])
    target_audience = st.selectbox("Target Audience", ["Kids", "Teens", "Adults"])

with col2:
    game_goal = st.text_input("Game Goal")
    art_style = st.selectbox("Art Style", ["Realistic", "Cartoon", "Pixel Art"])
    platform = st.multiselect("Target Platforms", ["PC", "Mobile", "Console"])

#添加详细偏好：
st.subheader("Detailed Preferences")
core_mechanics = st.multiselect(
    "Core Gameplay Mechanics",
    ["Combat", "Exploration", "Puzzle Solving"]
)
mood = st.multiselect(
    "Game Mood/Atmosphere",
    ["Epic", "Mysterious", "Peaceful"]
)
inspiration = st.text_area("Games for Inspiration")

#配置LLM设置：
llm_config = {
    "timeout": 600,
    "cache_seed": 44,
    "config_list": [{
        "model": "qwen-turbo",
        "api_key":'sk-b39fed46d35c45daa1eb1b0e8f087f43'
    }],
    "temperature": 0,
}

#创建任务代理：
task_agent = autogen.AssistantAgent(
    name="task_agent",
    llm_config=llm_config,
    system_message="You are a task provider. Your only job is to provide the task details to the other agents.",
)

#创建故事代理：
story_agent = autogen.AssistantAgent(
    name="story_agent",
    llm_config=llm_config,
    system_message="""
    You are a game story designer specializing in:
    1. Creating compelling narratives
    2. Designing memorable characters
    3. Developing game worlds
    4. Planning story progression
    """
)

#创建游戏代理：
gameplay_agent = autogen.AssistantAgent(
    name="gameplay_agent",
    llm_config=llm_config,
    system_message="""
    You are a game mechanics designer focusing on:
    1. Core gameplay loops
    2. Progression systems
    3. Player interactions
    4. Game balance
    """
)

#创建可视化代理：
visuals_agent = autogen.AssistantAgent(
    name="visuals_agent",
    llm_config=llm_config,
    system_message="""
    You are an art director responsible for:
    1. Visual style guides
    2. Character aesthetics
    3. Environmental design
    4. Audio direction
    """
)

#创建技术代理：
tech_agent = autogen.AssistantAgent(
    name="tech_agent",
    llm_config=llm_config,
    system_message="""
    You are a technical director handling:
    1. Game engine selection
    2. Technical requirements
    3. Development pipeline
    4. Performance optimization
    """
)

#实现顺序代理执行：
def run_agents_sequentially(task):
    task_agent.initiate_chat(story_agent, task)
    story_response = story_agent.last_message()["content"]

    task_agent.initiate_chat(gameplay_agent, task)
    gameplay_response = gameplay_agent.last_message()["content"]

    task_agent.initiate_chat(visuals_agent, task)
    visuals_response = gameplay_agent.last_message()["content"]

    task_agent.initiate_chat(tech_agent, task)
    tech_response = gameplay_agent.last_message()["content"]

    # Similar for visuals and tech agents
    return {
        "story": story_response,
        "gameplay": gameplay_response,
        "visuals": visuals_response,
        "tech": tech_response,
    }

#建立群聊协作：
groupchat = GroupChat(
    agents=[task_agent, story_agent, gameplay_agent,
            visuals_agent, tech_agent],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=5
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

#创建结果显示：
with st.expander("Story Design"):
    st.markdown(st.session_state.output['story'])

with st.expander("Gameplay Mechanics"):
    st.markdown(st.session_state.output['gameplay'])