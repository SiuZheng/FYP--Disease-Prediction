package com.example.as_heartdiseaseprediction.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalFocusManager
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import com.example.as_heartdiseaseprediction.model.healthQuestions
import com.example.as_heartdiseaseprediction.ui.components.QuestionCard
import com.example.as_heartdiseaseprediction.ui.viewmodel.PredictionViewModel
import androidx.compose.ui.graphics.Color

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GeneralUserFormScreen(
    onBackClick: () -> Unit,
    onSubmit: (PredictionRequest) -> Unit,
    viewModel: PredictionViewModel
) {
    var currentQuestionIndex by remember { mutableStateOf(0) }
    val answers = remember { mutableStateMapOf<Int, Any>() }
    var showCompletionDialog by remember { mutableStateOf(false) }
    val focusManager = LocalFocusManager.current
    
    // Clear state when entering the screen
    LaunchedEffect(Unit) {
        viewModel.clearState()
    }
    
    if (showCompletionDialog) {
        focusManager.clearFocus() // Clear any active focus when dialog shows
        AlertDialog(
            onDismissRequest = { /* Prevent dismiss on outside click */ },
            title = { Text("Questionnaire Complete") },
            text = { Text("What would you like to do?") },
            confirmButton = {
                Column(
                    modifier = Modifier.padding(horizontal = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Button(
                        onClick = {
                            // Log all answers for debugging
                            println("Debug - All answers: $answers")
                            
                            // Convert answers to PredictionRequest
                            val request = PredictionRequest(
                                Gender = when (answers[1]?.toString()) {
                                    "Male" -> 1
                                    "Female" -> 0
                                    else -> 0
                                },
                                Age = when (val ageValue = answers[2]) {
                                    is Float -> ageValue.toInt()
                                    is String -> ageValue.toIntOrNull() ?: 0
                                    else -> 0
                                },
                                Chest_Pain = if (answers[3] == true) 1 else 0,
                                Shortness_of_Breath = if (answers[4] == true) 1 else 0,
                                Fatigue = if (answers[5] == true) 1 else 0,
                                Palpitations = if (answers[6] == true) 1 else 0,
                                Dizziness = if (answers[7] == true) 1 else 0,
                                Swelling = if (answers[8] == true) 1 else 0,
                                Pain_Arms_Jaw_Back = if (answers[9] == true) 1 else 0,
                                Cold_Sweats_Nausea = if (answers[10] == true) 1 else 0,
                                Smoking = if (answers[11] == true) 1 else 0,
                                Obesity = if (answers[12] == true) 1 else 0,
                                Sedentary_Lifestyle = if (answers[13] == true) 1 else 0,
                                Family_History = if (answers[14] == true) 1 else 0,
                                Chronic_Stress = if (answers[15] == true) 1 else 0,
                                High_BP = if (answers[16] == true) 1 else 0,
                                High_Cholesterol = if (answers[17] == true) 1 else 0,
                                Diabetes = if (answers[18] == true) 1 else 0
                            )
                            // Set the username in the ViewModel
                            val userName = answers[0]?.toString() ?: "User"
                            println("Debug - GeneralUserFormScreen: Setting username to: $userName")
                            viewModel.setUsername(userName)
                            println("Debug - GeneralUserFormScreen: Final request: $request")
                            onSubmit(request)
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Submit")
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { 
                            showCompletionDialog = false
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Back")
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = { 
                            showCompletionDialog = false
                            currentQuestionIndex = 0
                            answers.clear()
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Restart")
                    }
                }
            }
        )
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Health Questionnaire") },
                navigationIcon = {
                    IconButton(
                        onClick = onBackClick,
                        enabled = !showCompletionDialog
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFFE6E0E9),
                    titleContentColor = MaterialTheme.colorScheme.onSurface,
                    navigationIconContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Progress indicator
            LinearProgressIndicator(
                progress = (currentQuestionIndex + (if (answers.containsKey(currentQuestionIndex)) 1 else 0)).toFloat() / healthQuestions.size,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            )
            
            Text(
                text = "Question ${currentQuestionIndex + 1} of ${healthQuestions.size}",
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyMedium
            )

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                if (currentQuestionIndex < healthQuestions.size) {
                    QuestionCard(
                        question = healthQuestions[currentQuestionIndex],
                        onAnswer = { answer ->
                            if (!showCompletionDialog) {
                                // Log the answer being stored
                                println("Debug - Storing answer for question ${currentQuestionIndex + 1}: $answer")
                                answers[currentQuestionIndex] = answer
                                if (currentQuestionIndex < healthQuestions.size - 1) {
                                    currentQuestionIndex++
                                } else {
                                    showCompletionDialog = true
                                }
                            }
                        },
                        onBack = {
                            if (!showCompletionDialog) {  // Only allow back navigation if dialog is not shown
                                if (currentQuestionIndex > 0) {
                                    currentQuestionIndex--
                                }
                            }
                        },
                        showBackButton = currentQuestionIndex > 0
                    )
                }
            }
        }
    }
} 