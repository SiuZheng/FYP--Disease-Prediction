package com.example.as_heartdiseaseprediction.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.example.as_heartdiseaseprediction.data.model.HealthcarePredictionRequest
import com.example.as_heartdiseaseprediction.ui.viewmodel.PredictionViewModel
import androidx.compose.ui.graphics.Color

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HealthcareProfessionalFormScreen(
    onBackClick: () -> Unit,
    onSubmit: (HealthcarePredictionRequest) -> Unit,
    viewModel: PredictionViewModel
) {
    var name by remember { mutableStateOf("") }
    var age by remember { mutableStateOf(TextFieldValue("")) }
    var sex by remember { mutableStateOf("") }
    var sexExpanded by remember { mutableStateOf(false) }
    var chestPainType by remember { mutableStateOf("") }
    var chestPainTypeExpanded by remember { mutableStateOf(false) }
    var restingBP by remember { mutableStateOf(TextFieldValue("")) }
    var fastingBS by remember { mutableStateOf(TextFieldValue("")) }
    var cholesterol by remember { mutableStateOf(TextFieldValue("")) }
    var restingECG by remember { mutableStateOf("") }
    var restingECGExpanded by remember { mutableStateOf(false) }
    var maxHR by remember { mutableStateOf(TextFieldValue("")) }
    var exerciseAngina by remember { mutableStateOf("") }
    var exerciseAnginaExpanded by remember { mutableStateOf(false) }
    var oldpeak by remember { mutableStateOf(TextFieldValue("")) }
    var stSlope by remember { mutableStateOf("") }
    var stSlopeExpanded by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Healthcare Professional Form") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
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
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // Name (not sent to API)
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text("Name") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Age
            OutlinedTextField(
                value = age,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        age = it
                    }
                },
                label = { Text("Age") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Sex
            ExposedDropdownMenuBox(
                expanded = sexExpanded,
                onExpandedChange = { sexExpanded = it }
            ) {
                OutlinedTextField(
                    value = sex,
                    onValueChange = { },
                    label = { Text("Sex") },
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = sexExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = sexExpanded,
                    onDismissRequest = { sexExpanded = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Male (M)") },
                        onClick = { 
                            sex = "M"
                            sexExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Female (F)") },
                        onClick = { 
                            sex = "F"
                            sexExpanded = false
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Chest Pain Type
            ExposedDropdownMenuBox(
                expanded = chestPainTypeExpanded,
                onExpandedChange = { chestPainTypeExpanded = it }
            ) {
                OutlinedTextField(
                    value = chestPainType,
                    onValueChange = { },
                    label = { Text("Chest Pain Type") },
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = chestPainTypeExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = chestPainTypeExpanded,
                    onDismissRequest = { chestPainTypeExpanded = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Typical Angina") },
                        onClick = { 
                            chestPainType = "TA"
                            chestPainTypeExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Atypical Angina") },
                        onClick = { 
                            chestPainType = "ATA"
                            chestPainTypeExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Non-anginal Pain") },
                        onClick = { 
                            chestPainType = "NAP"
                            chestPainTypeExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Asymptomatic") },
                        onClick = { 
                            chestPainType = "ASY"
                            chestPainTypeExpanded = false
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Resting Blood Pressure
            OutlinedTextField(
                value = restingBP,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        restingBP = it
                    }
                },
                label = { Text("Resting Blood Pressure (mm HG)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Fasting Blood Sugar
            OutlinedTextField(
                value = fastingBS,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        fastingBS = it
                    }
                },
                label = { Text("Fasting Blood Sugar (mg/dl)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Cholesterol
            OutlinedTextField(
                value = cholesterol,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        cholesterol = it
                    }
                },
                label = { Text("Cholesterol (mg/dl)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Resting ECG
            ExposedDropdownMenuBox(
                expanded = restingECGExpanded,
                onExpandedChange = { restingECGExpanded = it }
            ) {
                OutlinedTextField(
                    value = restingECG,
                    onValueChange = { },
                    label = { Text("Resting ECG Results") },
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = restingECGExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = restingECGExpanded,
                    onDismissRequest = { restingECGExpanded = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Normal") },
                        onClick = { 
                            restingECG = "Normal"
                            restingECGExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("ST-T Wave Abnormality") },
                        onClick = { 
                            restingECG = "ST"
                            restingECGExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Left Ventricular Hypertrophy") },
                        onClick = { 
                            restingECG = "LVH"
                            restingECGExpanded = false
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Max Heart Rate
            OutlinedTextField(
                value = maxHR,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        maxHR = it
                    }
                },
                label = { Text("Max Heart Rate") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Exercise Induced Angina
            ExposedDropdownMenuBox(
                expanded = exerciseAnginaExpanded,
                onExpandedChange = { exerciseAnginaExpanded = it }
            ) {
                OutlinedTextField(
                    value = exerciseAngina,
                    onValueChange = { },
                    label = { Text("Exercise Induced Angina") },
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = exerciseAnginaExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = exerciseAnginaExpanded,
                    onDismissRequest = { exerciseAnginaExpanded = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Yes") },
                        onClick = { 
                            exerciseAngina = "Y"
                            exerciseAnginaExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("No") },
                        onClick = { 
                            exerciseAngina = "N"
                            exerciseAnginaExpanded = false
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Oldpeak
            OutlinedTextField(
                value = oldpeak,
                onValueChange = { 
                    if (it.text.isEmpty() || it.text.all { char -> char.isDigit() }) {
                        oldpeak = it
                    }
                },
                label = { Text("Oldpeak") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // ST Slope
            ExposedDropdownMenuBox(
                expanded = stSlopeExpanded,
                onExpandedChange = { stSlopeExpanded = it }
            ) {
                OutlinedTextField(
                    value = stSlope,
                    onValueChange = { },
                    label = { Text("ST Slope") },
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = stSlopeExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = stSlopeExpanded,
                    onDismissRequest = { stSlopeExpanded = false }
                ) {
                    DropdownMenuItem(
                        text = { Text("Up Sloping") },
                        onClick = { 
                            stSlope = "Up"
                            stSlopeExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Flat") },
                        onClick = { 
                            stSlope = "Flat"
                            stSlopeExpanded = false
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Down Sloping") },
                        onClick = { 
                            stSlope = "Down"
                            stSlopeExpanded = false
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Submit Button
            Button(
                onClick = {
                    // Set the username in the ViewModel
                    viewModel.setUsername(name)
                    
                    // Create and submit the request
                    val request = HealthcarePredictionRequest(
                        Age = age.text.toIntOrNull() ?: 0,
                        Sex = sex,
                        ChestPainType = chestPainType,
                        RestingBP = restingBP.text.toIntOrNull() ?: 0,
                        FastingBS = if (fastingBS.text.toIntOrNull() ?: 0 > 120) 1 else 0,
                        Cholesterol = cholesterol.text.toIntOrNull() ?: 0,
                        RestingECG = restingECG,
                        MaxHR = maxHR.text.toIntOrNull() ?: 0,
                        ExerciseAngina = exerciseAngina,
                        Oldpeak = oldpeak.text.toIntOrNull() ?: 0,
                        ST_Slope = stSlope
                    )
                    onSubmit(request)
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Submit")
            }
        }
    }
} 