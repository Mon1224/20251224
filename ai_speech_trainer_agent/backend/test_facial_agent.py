from agents.facial_expression_agent import facial_expression_agent

if __name__ == "__main__":
    print("=== testing facial_expression_agent ===")

    response = facial_expression_agent.run(
        input={
            "video_path": r"C:\Users\Lenovo\Downloads\Vedio_trim.mp4"
        }
    )

    print("\n=== RAW RESPONSE OBJECT ===")
    print(response)

    print("\n=== RESPONSE CONTENT ===")
    print(response.content)
