from langchain_core.messages import HumanMessage, SystemMessage


def build_messages(user_question: str, context: str, metadata_intent: bool) -> list[SystemMessage | HumanMessage]:
    if metadata_intent:
        system_prompt = (
            "Ты помощник по договору. Отвечай только на русском языке и только по "
            "переданному контексту. Для вопросов про номер и дату называй только те "
            "реквизиты, которые относятся к одному и тому же договору."
        )
        human_prompt = (
            f"Контекст документа:\n{context}\n\n"
            f"Вопрос пользователя:\n{user_question}\n\n"
            "Ответь коротко, в 1-3 строках. Отвечай только на то, что прямо "
            "спрошено. Дата должна соответствовать номеру договора. Если дата есть "
            "только в приложении к этому же договору, напиши это явно."
        )
    else:
        system_prompt = (
            "Ты помощник по вопросам к документу. Отвечай только на русском языке и "
            "только на основе переданного контекста. Если в контексте нет данных для "
            "точного ответа, прямо скажи, что информации недостаточно."
        )
        human_prompt = (
            f"Контекст документа:\n{context}\n\n"
            f"Вопрос пользователя:\n{user_question}\n\n"
            "Сформируй короткий и точный ответ."
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
