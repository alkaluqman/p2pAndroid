package com.example.applayout.LocalAssets

import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Data.Model.Model

@Composable
fun EditFinetuningRelationshipDialog(
    onDismiss: () -> Unit,
    models: List<Model>,
    dataset: List<Dataset>,
    onSubmit: (String, String, String, Int, Int) -> Unit
) {

    var selectedModel by remember { mutableStateOf<Model?>(null) }
    var selectedDataset by remember { mutableStateOf<Dataset?>(null) }
    var expandedModel by remember { mutableStateOf(false) }
    var expandedDataset by remember { mutableStateOf(false) }
    var numEpochs by remember { mutableStateOf("100") }
    var batchSize by remember { mutableStateOf("10") }
    var newModelName by remember { mutableStateOf("") }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(dismissOnBackPress = true, dismissOnClickOutside = true)
    ) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.padding(16.dp),
            tonalElevation = 4.dp
        ) {
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState())

            ) {
                Text(
                    "Edit Relationship",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                Text("Model Unique Identifier")
                Box {
                    Text(text = selectedModel?.uniqueIdentifier ?: "Select Model",
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedModel = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary))
                    DropdownMenu(expanded = expandedModel,
                        onDismissRequest = { expandedModel = false }) {
                        models.forEach { model ->
                            DropdownMenuItem(text = { Text(model.uniqueIdentifier) }, onClick = {
                                selectedModel = model
                                expandedModel = false
                            })
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text("Source Unique Identifiers")
                Box {
                    Text(text = selectedDataset?.uniqueIdentifier ?: "Select Source Dataset",
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedDataset = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary))
                    DropdownMenu(expanded = expandedDataset,
                        onDismissRequest = { expandedDataset = false }) {
                        dataset.forEach { data ->
                            DropdownMenuItem(text = {
                                Text(data.uniqueIdentifier)
                            }, onClick = {
                                selectedDataset = data
                                expandedDataset = false
                            })
                        }
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))

                Box {
                    TextField(
                        value = newModelName,
                        onValueChange = { newModelName = it },
                        label = { Text("New Model File Name") },
                        placeholder = { Text("Leave blank for a random file name") },
                        singleLine = true
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))

                Box {
                    NumberInputField(value = numEpochs,
                        labelText = "Number of Epochs",
                        onValueChange = {
                            numEpochs = it
                        })
                }
                Box {
                    NumberInputField(value = batchSize,
                        labelText = "Batch Size",
                        onValueChange = {
                            batchSize = it
                        })
                }


                Button(
                    onClick = {
                        onSubmit(
                            selectedModel!!.uniqueIdentifier,
                            selectedDataset!!.uniqueIdentifier,
                            newModelName,
                            numEpochs.toInt(),
                            batchSize.toInt()
                        )
                    },
                    enabled = selectedModel != null && selectedDataset != null && listOf(
                        numEpochs,
                        batchSize
                    ).all { it.isNotBlank() && it.toIntOrNull()!! > 0 },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Submit")
                }
            }
        }
    }
}