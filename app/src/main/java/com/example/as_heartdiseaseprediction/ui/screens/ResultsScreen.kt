package com.example.as_heartdiseaseprediction.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.example.as_heartdiseaseprediction.data.model.PredictionResponse
import android.graphics.BitmapFactory
import android.util.Base64
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.ui.viewinterop.AndroidView // <- Important for AndroidView
import android.widget.TextView                   // <- To create TextView manually
import io.noties.markwon.Markwon                 // <- For Markwon markdown rendering
import androidx.compose.ui.graphics.Color
import com.example.as_heartdiseaseprediction.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsScreen(
    predictionResult: Result<PredictionResponse>?,
    isLoading: Boolean,
    onBackClick: () -> Unit,
    username: String,
    shapPlot: String?,
    explanation: String?,
    onAboutClick: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    var showShapPlot by remember { mutableStateOf(false) }
    var shapPlotImage by remember { mutableStateOf<ImageBitmap?>(null) }

    // Convert base64 to ImageBitmap when shapPlot changes
    LaunchedEffect(shapPlot) {
        shapPlot?.let { base64String ->
            try {
                val imageBytes = Base64.decode(base64String, Base64.DEFAULT)
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                shapPlotImage = bitmap.asImageBitmap()
            } catch (e: Exception) {
                println("Error converting SHAP plot: ${e.message}")
            }
        }
    }

    if (showShapPlot) {
        Dialog(onDismissRequest = { showShapPlot = false }) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(0.8f)
                    .padding(16.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp)
                ) {
                    // Close button
                    Box(
                        modifier = Modifier.fillMaxWidth(),
                        contentAlignment = Alignment.TopEnd
                    ) {
                        IconButton(
                            onClick = { showShapPlot = false },
                            modifier = Modifier.padding(8.dp)
                        ) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Close",
                                tint = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }

                    // SHAP plot image
                    shapPlotImage?.let { image ->
                        Image(
                            bitmap = image,
                            contentDescription = "SHAP Plot",
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(0.5f)
                        )
                    } ?: run {
                        Text(
                            text = "Unable to display SHAP plot",
                            modifier = Modifier
                                .align(Alignment.CenterHorizontally)
                                .padding(16.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Explanation text
                    explanation?.let { text ->
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(0.5f)
                                .verticalScroll(rememberScrollState())
                        ) {
                            AndroidView(
                                factory = { context ->
                                    TextView(context).apply {
                                        // Create Markwon instance
                                        val markwon = Markwon.create(context)

                                        // Set the markdown text
                                        markwon.setMarkdown(this, text)

                                        // Optional styling
                                        setPadding(32, 0, 32, 0)
                                        textSize = 18f
                                        setTextColor(android.graphics.Color.parseColor("#333333"))
                                    }
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                            )
                        }
                    }
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Prediction Results") },
                navigationIcon = {
                    IconButton(
                        onClick = onBackClick
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to Home")
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
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            when {
                isLoading -> {
                    CircularProgressIndicator()
                }
                predictionResult == null -> {
                    Text("No prediction results available")
                }
                predictionResult.isSuccess -> {
                    val response = predictionResult.getOrNull()
                    Card(
                        modifier = Modifier
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
                            Text(
                                text = "Prediction Result",
                                style = MaterialTheme.typography.headlineMedium,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(24.dp))

                            Text(
                                text = "Hello, $username",
                                style = MaterialTheme.typography.titleLarge,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(16.dp))

                            Text(
                                text = "Your probability of having a heart disease is",
                                style = MaterialTheme.typography.titleMedium,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            val probabilityPercent = (response?.probability ?: 0f) * 100
                            val probabilityColor = when {
                                probabilityPercent < 30f -> Color(0xFF00E676) // Green
                                probabilityPercent < 60f -> Color(0xFFFFC107) // Orange
                                else -> Color(0xFFFF1744) // Red
                            }
                            Text(
                                text = String.format("%.2f%%", probabilityPercent),
                                style = MaterialTheme.typography.displayMedium,
                                color = probabilityColor,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(16.dp))

                            // Why we predicted this button
                            Button(
                                onClick = { showShapPlot = true },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Why we predicted this")
                            }
                        }
                    }
                    
                    // Disclaimer notice
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Row(
                            modifier = Modifier
                                .padding(12.dp)
                                .fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.primary
                            )
                            
                            Spacer(modifier = Modifier.width(8.dp))
                            
                            Text(
                                text = stringResource(R.string.prediction_notice),
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.weight(1f)
                            )
                            
                            TextButton(
                                onClick = onAboutClick,
                                modifier = Modifier.padding(start = 8.dp)
                            ) {
                                Text(
                                    text = stringResource(R.string.prediction_notice_see_about),
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }
                predictionResult.isFailure -> {
                    Text(
                        text = "Error: ${predictionResult.exceptionOrNull()?.message ?: "Unknown error"}",
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        }
    }
} 