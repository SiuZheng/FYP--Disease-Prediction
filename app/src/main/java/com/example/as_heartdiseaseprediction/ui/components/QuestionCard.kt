package com.example.as_heartdiseaseprediction.ui.components

import androidx.compose.animation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.as_heartdiseaseprediction.model.AnswerType
import com.example.as_heartdiseaseprediction.model.Question

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuestionCard(
    question: Question,
    onAnswer: (Any) -> Unit,
    onBack: () -> Unit,
    showBackButton: Boolean,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .padding(24.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Icon based on question type
            Icon(
                imageVector = when (question.icon) {
                    "heart" -> Icons.Filled.Favorite
                    "gender" -> Icons.Filled.Person
                    else -> Icons.Filled.Person
                },
                contentDescription = null,
                modifier = Modifier
                    .size(48.dp)
                    .padding(bottom = 16.dp),
                tint = MaterialTheme.colorScheme.primary
            )

            // Question text
            Text(
                text = question.question,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // Answer section based on question type
            when (question.answerType) {
                AnswerType.YesNo -> {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Button(
                            onClick = { onAnswer(true) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 8.dp)
                        ) {
                            Text("Yes")
                        }
                        Button(
                            onClick = { onAnswer(false) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("No")
                        }
                        
                        if (showBackButton) {
                            Spacer(modifier = Modifier.height(16.dp))
                            OutlinedButton(
                                onClick = onBack,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Back")
                            }
                        }
                    }
                }
                AnswerType.Gender -> {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Button(
                            onClick = { onAnswer("Male") },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Male")
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = { onAnswer("Female") },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Female")
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = { onAnswer("Other") },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Other")
                        }
                        
                        if (showBackButton) {
                            Spacer(modifier = Modifier.height(16.dp))
                            OutlinedButton(
                                onClick = onBack,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Back")
                            }
                        }
                    }
                }
                AnswerType.NumberInput -> {
                    var value by remember { mutableStateOf("") }
                    var isValid by remember { mutableStateOf(false) }
                    
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        OutlinedTextField(
                            value = value,
                            onValueChange = { 
                                if (it.isEmpty() || it.toFloatOrNull() != null) {
                                    value = it
                                    isValid = it.toFloatOrNull()?.let { number ->
                                        number in (question.minValue ?: 0f)..(question.maxValue ?: Float.MAX_VALUE)
                                    } ?: false
                                }
                            },
                            label = { Text("Enter ${question.unit ?: "value"}") },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.fillMaxWidth()
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Button(
                            onClick = { 
                                value.toFloatOrNull()?.let { number ->
                                    if (number in (question.minValue ?: 0f)..(question.maxValue ?: Float.MAX_VALUE)) {
                                        onAnswer(number)
                                    }
                                }
                            },
                            enabled = isValid,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Confirm")
                        }
                        
                        if (showBackButton) {
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedButton(
                                onClick = onBack,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Back")
                            }
                        }
                    }
                }
                AnswerType.Slider -> {
                    var sliderValue by remember { mutableStateOf(0f) }
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = sliderValue.toInt().toString(),
                            style = MaterialTheme.typography.bodyLarge
                        )
                        Slider(
                            value = sliderValue,
                            onValueChange = {
                                sliderValue = it
                            },
                            valueRange = (question.minValue ?: 0f)..(question.maxValue ?: 100f)
                        )
                        
                        Button(
                            onClick = { onAnswer(sliderValue) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Confirm")
                        }
                        
                        if (showBackButton) {
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedButton(
                                onClick = onBack,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Back")
                            }
                        }
                    }
                }
                AnswerType.Text -> {
                    var value by remember { mutableStateOf("") }
                    var isValid by remember { mutableStateOf(false) }
                    
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        OutlinedTextField(
                            value = value,
                            onValueChange = { 
                                value = it
                                isValid = it.isNotBlank()
                            },
                            label = { Text("Enter your name") },
                            modifier = Modifier.fillMaxWidth()
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Button(
                            onClick = { if (isValid) onAnswer(value) },
                            enabled = isValid,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Confirm")
                        }
                        
                        if (showBackButton) {
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedButton(
                                onClick = onBack,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Back")
                            }
                        }
                    }
                }
            }
        }
    }
} 