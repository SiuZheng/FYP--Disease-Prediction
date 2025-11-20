package com.example.as_heartdiseaseprediction.ui.screens

import androidx.compose.animation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.text.font.FontWeight
import com.example.as_heartdiseaseprediction.data.model.HealthcarePredictionRequest
import com.example.as_heartdiseaseprediction.data.model.PredictionEntry
import com.example.as_heartdiseaseprediction.data.model.PredictionRequest
import java.text.SimpleDateFormat
import java.util.*
import androidx.compose.ui.unit.sp



fun toYesNo(value: Any?): String {
    return when (value) {
        1, "1", true, "true" -> "Yes"
        0, "0", false, "false" -> "No"
        else -> value.toString()
    }
}

fun getRestingECGLabel(code: String): String {
    return when (code) {
        "Normal" -> "Normal"
        "ST" -> "ST-T Wave Abnormality"
        "LVH" -> "Left Ventricular Hypertrophy"
        else -> "Unknown"
    }
}

fun getChestPainTypeLabel(code: String): String {
    return when (code) {
        "TA" -> "Typical Angina"
        "ATA" -> "Atypical Angina"
        "NAP" -> "Non-anginal Pain"
        "ASY" -> "Asymptomatic"
        else -> "Unknown"
    }
}

fun getFastingBSLabel(value: Int): String {
    return if (value == 1) "> 120 mg/dL" else "<= 120 mg/dL"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    modifier: Modifier = Modifier,
    history: List<PredictionEntry>,
    isLoading: Boolean,
    onLoadHistory: () -> Unit,
    onBackClick: () -> Unit = {}
) {
    var expandedFilter by remember { mutableStateOf(false) }
    var expandedArrange by remember { mutableStateOf(false) }
    var selectedFilter by remember { mutableStateOf("All") }
    var selectedArrange by remember { mutableStateOf("By Time") }
    val filteredHistory = history.filter { entry ->
        when (selectedFilter) {
            "Healthcare Professional" -> entry.data is HealthcarePredictionRequest
            "General User" -> entry.data is PredictionRequest
            else -> true // All
        }
    }
    val sortedHistory = when (selectedArrange) {
        "By Name" -> filteredHistory.sortedBy { it.userName.lowercase() }
        "By Probability" -> filteredHistory.sortedByDescending { it.probability }
        else -> filteredHistory.sortedByDescending { it.timestamp } // Default: By Time (latest first)
    }
    LaunchedEffect(Unit) {
        onLoadHistory()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        text = "Prediction History",
                        style = MaterialTheme.typography.headlineSmall
                    ) 
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back to Home"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFFE6E0E9),
                    titleContentColor = MaterialTheme.colorScheme.onTertiary,
                    navigationIconContentColor = MaterialTheme.colorScheme.onTertiary
                )
            )
        }
    ) { padding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {

        // Filter and Arrange buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Filter Dropdown
            Box(
                modifier = Modifier.weight(1f)
            ) {
                Button(
                    onClick = { expandedFilter = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Filter: $selectedFilter")
                    Icon(
                        imageVector = if (expandedFilter) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                        contentDescription = null
                    )
                }
                DropdownMenu(
                    expanded = expandedFilter,
                    onDismissRequest = { expandedFilter = false }
                ) {
                    listOf("All", "Healthcare Professional", "General User").forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option) },
                            onClick = {
                                selectedFilter = option
                                expandedFilter = false
                            }
                        )
                    }
                }
            }

            // Arrange Dropdown
            Box(
                modifier = Modifier.weight(1f)
            ) {
                Button(
                    onClick = { expandedArrange = true },
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = "Arrange: $selectedArrange",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Icon(
                        imageVector = if (expandedArrange) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                        contentDescription = null
                    )
                }
                DropdownMenu(
                    expanded = expandedArrange,
                    onDismissRequest = { expandedArrange = false }
                ) {
                    listOf("By Time", "By Name", "By Probability").forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option) },
                            onClick = {
                                selectedArrange = option
                                expandedArrange = false
                            }
                        )
                    }
                }
            }
        }

        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    CircularProgressIndicator()
                    Text(
                        text = "Loading prediction history...",
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
        } else if (sortedHistory.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "No prediction history available",
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(sortedHistory) { entry ->
                    PredictionHistoryCard(entry = entry)
                }
            }
        }
        }
    }
}

@Composable
fun PredictionHistoryCard(entry: PredictionEntry) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            // Header section
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = entry.userName,
                        style = MaterialTheme.typography.headlineSmall.copy(
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.sp
                        ),
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                    Text(
                        text = formatTimestamp(entry.timestamp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = String.format("%.1f%%", entry.probability * 100),
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontWeight = FontWeight.Bold ,
                            fontSize = 32.sp
                        ),
                        color = getProbabilityColor(entry.probability)
                    )
                    IconButton(onClick = { expanded = !expanded }) {
                        Icon(
                            imageVector = if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = if (expanded) "Collapse" else "Expand"
                        )
                    }
                }
            }

            // Prediction type
            Text(
                text = when (entry.data) {
                    is PredictionRequest -> "General User Prediction"
                    is HealthcarePredictionRequest -> "Healthcare Professional Prediction"
                    else -> "Unknown Prediction Type"
                },
                style = MaterialTheme.typography.labelMedium.copy(
                    fontSize = 15.sp, // Make font bigger
                    fontWeight = FontWeight.Medium // Optional for emphasis
                ),
                color = when (entry.data) {
                    is HealthcarePredictionRequest -> Color(0xFF2196F3)
                    else -> MaterialTheme.colorScheme.primary
                },
                modifier = Modifier.padding(top = 8.dp)
            )

            // Detailed information
            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp)
                ) {
                    when (val data = entry.data) {
                        is PredictionRequest -> {
                            PredictionRequestDetails(data)
                        }
                        is HealthcarePredictionRequest -> {
                            HealthcarePredictionRequestDetails(data)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun PredictionRequestDetails(request: PredictionRequest) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        DetailRow("Gender", if (request.Gender == 1) "Male" else "Female")
        DetailRow("Age", request.Age.toString())
        DetailRow("Chest Pain", toYesNo(request.Chest_Pain))
        DetailRow("Shortness of Breath", toYesNo(request.Shortness_of_Breath))
        DetailRow("Fatigue", toYesNo(request.Fatigue))
        DetailRow("Palpitations", toYesNo(request.Palpitations))
        DetailRow("Dizziness", toYesNo(request.Dizziness))
        DetailRow("Swelling", toYesNo(request.Swelling))
        DetailRow("Pain (Arms, Jaw, Back)",toYesNo( request.Pain_Arms_Jaw_Back))
        DetailRow("Cold Sweats/Nausea", toYesNo(request.Cold_Sweats_Nausea))
        DetailRow("Smoking", toYesNo(request.Smoking))
        DetailRow("Obesity", toYesNo(request.Obesity))
        DetailRow("Sedentary Lifestyle", toYesNo(request.Sedentary_Lifestyle))
        DetailRow("Family History", toYesNo(request.Family_History))
        DetailRow("Chronic Stress", toYesNo(request.Chronic_Stress))
        DetailRow("High BP", toYesNo(request.High_BP))
        DetailRow("High Cholesterol", toYesNo(request.High_Cholesterol))
        DetailRow("Diabetes", toYesNo(request.Diabetes))
    }
}


@Composable
fun HealthcarePredictionRequestDetails(request: HealthcarePredictionRequest) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        DetailRow("Age", request.Age.toString())
        DetailRow("Sex", if (request.Sex == "M") "Male (M)" else "Female (F)")
        DetailRow("Chest Pain Type", getChestPainTypeLabel(request.ChestPainType))
        DetailRow("Resting BP", request.RestingBP.toString())
        DetailRow("Fasting BS", getFastingBSLabel(request.FastingBS))
        DetailRow("Cholesterol", request.Cholesterol.toString())
        DetailRow("Resting ECG", getRestingECGLabel(request.RestingECG))
        DetailRow("Max HR", request.MaxHR.toString())
        DetailRow("Exercise Angina", if (request.ExerciseAngina == "Y") "Yes" else "No")
        DetailRow("Oldpeak", request.Oldpeak.toString())
        DetailRow("ST Slope", request.ST_Slope)
    }
}

@Composable
fun DetailRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium.copy(
                fontWeight = FontWeight.Bold
            ),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

private fun formatTimestamp(timestamp: Long): String {
    val date = Date(timestamp)
    val formatter = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
    return formatter.format(date)
}

@Composable
private fun getProbabilityColor(probability: Float): Color {
    val percentage = probability * 100
    return when {
        percentage < 30 -> Color(0xFF00E676) // Green
        percentage < 60 -> Color(0xFFFFC107) // Amber
        else -> Color(0xFFFF1744) // Red
    }
}