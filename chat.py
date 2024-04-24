from main import generation
while True:
    try:
        user_input=input("输入对话内容")
        generation(user_input)
    except:
        continue