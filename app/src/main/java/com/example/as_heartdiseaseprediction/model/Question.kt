package com.example.as_heartdiseaseprediction.model

sealed class AnswerType {
    object YesNo : AnswerType()
    object Gender : AnswerType()
    object NumberInput : AnswerType()
    object Slider : AnswerType()
    object Text : AnswerType()
}

data class Question(
    val id: Int,
    val question: String,
    val answerType: AnswerType,
    val icon: String,
    val minValue: Float? = null,
    val maxValue: Float? = null,
    val unit: String? = null
)

val healthQuestions = listOf(
    Question(1, "What is your name?", AnswerType.Text, "person"),
    Question(2, "What is your gender?", AnswerType.Gender, "gender"),
    Question(3, "What is your age?", AnswerType.NumberInput, "age", 0f, 120f, "years"),
    Question(4, "Do you experience chest pain?", AnswerType.YesNo, "heart"),
    Question(5, "Have you had shortness of breath recently?", AnswerType.YesNo, "lungs"),
    Question(6, "Are you often fatigued?", AnswerType.YesNo, "tired"),
    Question(7, "Do you feel heart palpitations?", AnswerType.YesNo, "heartbeat"),
    Question(8, "Have you experienced dizziness?", AnswerType.YesNo, "dizzy"),
    Question(9, "Do you have any swelling in your legs, ankles, or feet?", AnswerType.YesNo, "leg"),
    Question(10, "Do you feel pain in your arms, jaw, or back?", AnswerType.YesNo, "pain"),
    Question(11, "Have you had cold sweats or nausea?", AnswerType.YesNo, "sweat"),
    Question(12, "Are you a smoker?", AnswerType.YesNo, "smoking"),
    Question(13, "Are you overweight or obese?", AnswerType.YesNo, "weight"),
    Question(14, "Would you describe your lifestyle as mostly sedentary?", AnswerType.YesNo, "lifestyle"),
    Question(15, "Is there a family history of heart-related diseases?", AnswerType.YesNo, "family"),
    Question(16, "Do you experience chronic stress?", AnswerType.YesNo, "stress"),
    Question(17, "Do you have high blood pressure?", AnswerType.YesNo, "blood_pressure"),
    Question(18, "Do you have high cholesterol?", AnswerType.YesNo, "cholesterol"),
    Question(19, "Have you been diagnosed with diabetes?", AnswerType.YesNo, "diabetes")
) 