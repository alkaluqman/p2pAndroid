package com.example.applayout.Marketplace

import android.util.Log
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.applayout.Data.Model.Model
import com.example.applayout.Data.Model.modelTasks
import com.example.applayout.LocalAssets.fetchModelOwner
import com.example.applayout.LocalAssets.listLocalResources
import com.example.applayout.Models.PublicModelCard
import kotlinx.coroutines.launch
import java.io.File


@Composable
fun MarketplaceScreen(filesDir: File, navController: NavController) {
    val scope = rememberCoroutineScope()
    var modelList by remember { mutableStateOf<List<Model>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var searchQuery by remember { mutableStateOf("") }
    var selectedTask by remember { mutableStateOf<String?>(null) }
    var expanded by remember { mutableStateOf(false) }
    var downloadedModels by remember { mutableStateOf<List<String>>(emptyList()) }
    fun fetchModels() {
        downloadedModels =
            listLocalResources(filesDir, "models", true).map { it.substringBeforeLast('.') }
        Log.d("marketplace", "downloadedModels: $downloadedModels")
        scope.launch {
            try {
                Log.d("marketplace", getMarketplaceFiles().toString())
                modelList = getMarketplaceFiles()
                    .filter { model ->
                        !fetchModelOwner(
                            model.uniqueIdentifier,
                            "alice"
                        ) && model.uniqueIdentifier !in downloadedModels
                    }
                    .sortedByDescending { model -> model.likes + model.usage }
                errorMessage = null
                Log.d("marketplace", "modelList: $modelList")
            } catch (e: Exception) {
                errorMessage = "Failed to load models. Please try again."
            } finally {
                isLoading = false
            }
        }
    }

    LaunchedEffect(Unit) {
        fetchModels()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                label = { Text("Search models") },
                modifier = Modifier
                    .weight(1f)
                    .padding(end = 8.dp)
            )
            Box {
                Button(
                    onClick = { expanded = true },
                    modifier = Modifier
                        .wrapContentSize()
                        .padding(end = 8.dp)
                ) {
                    Text(text = selectedTask ?: "All Tasks")
                }
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    DropdownMenuItem(text = { Text("All Tasks") }, onClick = {
                        selectedTask = null
                        expanded = false
                    })
                    modelTasks.forEach { task ->
                        DropdownMenuItem(text = { Text(task) }, onClick = {
                            selectedTask = task
                            expanded = false
                        })
                    }
                }
            }
        }

        when {
            isLoading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }

            !errorMessage.isNullOrEmpty() -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = errorMessage ?: "An error occurred",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }

            else -> {
                val filteredModels = modelList
                    .filter {
                        it.uniqueIdentifier.contains(searchQuery, ignoreCase = true) ||
                                it.architecture.contains(searchQuery, ignoreCase = true)
                    }
                    .filter { model ->
                        selectedTask == null || model.model_task == selectedTask
                    }

                LazyColumn {
                    items(filteredModels) { model ->
                        PublicModelCard(model, navController,
                            onDownload = {
                                scope.launch {
                                    downloadModelFile(
                                        model.public_link,
                                        filesDir,
                                        model.uniqueIdentifier
                                    )
                                    increment(model.uniqueIdentifier, true)
                                    fetchModels()
                                }
                            },
                            onLike = {
                                scope.launch {
                                    increment(model.uniqueIdentifier, false)
                                    fetchModels()
                                }
                            })
                    }
                }
            }
        }
    }
}