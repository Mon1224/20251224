from agents.tools.facial_expression_core import analyze_facial_expressions_core

if __name__ == "__main__":
    print("testing core logic...")

    result = analyze_facial_expressions_core(
        r"C:\Users\Lenovo\Downloads\衡水中学“学霸”励志演讲：这世间，唯有青春与梦想不可辜负.mp4"
    )

    print("result:")
    print(result)
