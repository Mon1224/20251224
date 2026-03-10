#创建游戏详情的用户界面：
import sys
import io

# 强制设置标准输出为 UTF-8，解决 ASCII 编码报错
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import streamlit as st
import autogen

# Initialize session state
if 'output' not in st.session_state:
    st.session_state.output = {
        'story': '', 'gameplay': '',
        'visuals': '', 'tech': ''
    }

# Sidebar for API key input
st.sidebar.title("API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Add guidance in sidebar
st.sidebar.success("""
✨ **Getting Started**

Please provide inputs and features for your dream game! Consider:
- The overall vibe and setting
- Core gameplay elements
- Target audience and platforms
- Visual style preferences
- Technical requirements

The AI agents will collaborate to develop a comprehensive game concept based on your specifications.
""")

#创建游戏详情的用户界面:Main app UI
st.title("🎮 AI Game Design Agent Team")

# Add agent information below title
st.info("""
**Meet Your AI Game Design Team:**

🎭 **Story Agent** - Crafts compelling narratives and rich worlds

🎮 **Gameplay Agent** - Creates engaging mechanics and systems

🎨 **Visuals Agent** - Shapes the artistic vision and style

⚙️ **Tech Agent** - Provides technical direction and solutions

These agents collaborate to create a comprehensive game concept based on your inputs.
""")

# User inputs
st.subheader("Game Details")
col1, col2 = st.columns(2)

with col1:
    background_vibe = st.text_input("Background Vibe","Epic fantasy with dragons")
    game_type = st.selectbox("Game Type", ["RPG", "Action", "Adventure", "Puzzle", "Strategy", "Simulation", "Platform", "Horror"])
    target_audience = st.selectbox("Target Audience", ["Kids (7-12)", "Teens (13-17)", "Young Adults (18-25)", "Adults (26+)", "All Ages"])
    player_perspective = st.selectbox("Player Perspective",
                                      ["First Person", "Third Person", "Top Down", "Side View", "Isometric"])
    multiplayer = st.selectbox("Multiplayer Support",
                               ["Single Player Only", "Local Co-op", "Online Multiplayer", "Both Local and Online"])

with col2:
    game_goal = st.text_input("Game Goal", "Save the kingdom from eternal winter")
    art_style = st.selectbox("Art Style",
                             ["Realistic", "Cartoon", "Pixel Art", "Stylized", "Low Poly", "Anime", "Hand-drawn"])
    platform = st.multiselect("Target Platforms",
                              ["PC", "Mobile", "PlayStation", "Xbox", "Nintendo Switch", "Web Browser"])
    development_time = st.slider("Development Time (months)", 1, 36, 12)
    cost = st.number_input("Budget (USD)", min_value=0, value=10000, step=5000)

#添加详细偏好：
st.subheader("Detailed Preferences")
col3, col4 = st.columns(2)

with col3:
    core_mechanics = st.multiselect(
        "Core Gameplay Mechanics",
        ["Combat", "Exploration", "Puzzle Solving", "Resource Management", "Base Building", "Stealth", "Racing", "Crafting"]
    )
    mood = st.multiselect(
        "Game Mood/Atmosphere",
        ["Epic", "Mysterious", "Peaceful", "Tense", "Humorous", "Dark", "Whimsical", "Scary"]
    )

with col4:
    inspiration = st.text_area("Games for Inspiration (comma-separated)", "")
    unique_features = st.text_area("Unique Features or Requirements", "")

depth = st.selectbox("Level of Detail in Response", ["Low", "Medium", "High"])

#配置LLM设置：
llm_config = {"config_list": [{"model": "qwen-turbo","api_key": api_key,"base_url":
    "https://dashscope.aliyuncs.com/compatible-mode/v1"}],"cache_seed": None}


# 统一定义 Agent
def get_agents():
    task_agent = autogen.AssistantAgent(name="task_agent", llm_config=llm_config,
                                        human_input_mode="NEVER", system_message="You are a task provider. Your only job is to provide the task details to the other agents.")

    story_agent = autogen.AssistantAgent(name="story_agent", llm_config=llm_config,
                                         system_message='''You are a game story designer specializing in:
                                         1. Creating compelling narratives
                                         2. Designing memorable characters
                                         3. Developing game worlds
                                         4. Planning story progression''')

    gameplay_agent = autogen.AssistantAgent(name="gameplay_agent", llm_config=llm_config,
                                            system_message="""You are a game mechanics designer focusing on:
                                            1. Core gameplay loops
                                            2. Progression systems
                                            3. Player interactions
                                            4. Game balance""")

    visuals_agent = autogen.AssistantAgent(name="visuals_agent", llm_config=llm_config,
                                           system_message="""You are an art director responsible for:
                                           1. Visual style guides
                                           2. Character aesthetics
                                           3. Environmental design
                                           4. Audio direction""")

    tech_agent = autogen.AssistantAgent(name="tech_agent", llm_config=llm_config,
                                        system_message="""You are a technical director handling:
                                        1. Game engine selection
                                        2. Technical requirements
                                        3. Development pipeline
                                        4. Performance optimization""")

    return task_agent, story_agent, gameplay_agent, visuals_agent, tech_agent

# Button to start the agent collaboration
if st.button("Generate Game Concept"):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        task = f"""
                    Create a game concept with the following details:
                    - Background Vibe: {background_vibe}
                    - Game Type: {game_type}
                    - Game Goal: {game_goal}
                    - Target Audience: {target_audience}
                    - Player Perspective: {player_perspective}
                    - Multiplayer Support: {multiplayer}
                    - Art Style: {art_style}
                    - Target Platforms: {', '.join(platform)}
                    - Development Time: {development_time} months
                    - Budget: ${cost:,}
                    - Core Mechanics: {', '.join(core_mechanics)}
                    - Mood/Atmosphere: {', '.join(mood)}
                    - Inspiration: {inspiration}
                    - Unique Features: {unique_features}
                    - Detail Level: {depth}
                    """
        try:
            task_agent, story_agent, gameplay_agent, visuals_agent, tech_agent = get_agents()
            # 使用 st.status 展示工作流
            with st.spinner('🤖 AI Agents are collaborating on your game concept...') as status:
                st.write("🎭 正在构思世界观故事...")
                task_agent.initiate_chat(story_agent, message=task, max_turns=1, silent=True)
                st.session_state.output['story'] = story_agent.last_message()["content"]

                st.write("🎮 正在设计核心玩法...")
                task_agent.initiate_chat(gameplay_agent, message=task, max_turns=1, silent=True)
                st.session_state.output['gameplay'] = gameplay_agent.last_message()["content"]

                st.write("🎨 正在规划视觉表现...")
                task_agent.initiate_chat(visuals_agent, message=task, max_turns=1, silent=True)
                st.session_state.output['visuals'] = visuals_agent.last_message()["content"]

                st.write("⚙️ 正在评估技术可行性...")
                task_agent.initiate_chat(tech_agent, message=task, max_turns=1, silent=True)
                st.session_state.output['tech'] = tech_agent.last_message()["content"]

                if status is not None:
                    status.update(label="✨ 策划方案生成完毕！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"发生错误: {str(e)}")


#创建结果显示：
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["📜 故事剧情", "🕹️ 玩法机制", "🖼️ 视觉美术", "💻 技术架构"])

with tab1:
    st.markdown(st.session_state.output['story'] or "暂无内容，请点击生成。")
with tab2:
    st.markdown(st.session_state.output['gameplay'] or "暂无内容，请点击生成。")
with tab3:
    st.markdown(st.session_state.output['visuals'] or "暂无内容，请点击生成。")
with tab4:
    st.markdown(st.session_state.output['tech'] or "暂无内容，请点击生成。")