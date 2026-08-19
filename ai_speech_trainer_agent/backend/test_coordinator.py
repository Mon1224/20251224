from agents.coordinator_agent import coordinator_agent

response = coordinator_agent.run({
        "role": "user",
        "content": {
            "video_path": r"C:\Users\Lenovo\Downloads\Vedio_trim.mp4"
        }
    })

print(response)
