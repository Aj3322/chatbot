while True:
    user_input = input("What problems are you facing? (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response_text = get_gemini_response(user_input+" keep the answer short and to the point. Only suggest the preventions")
    display(to_markdown(response_text + '\n\nIs there anything else I can help you with?'))

