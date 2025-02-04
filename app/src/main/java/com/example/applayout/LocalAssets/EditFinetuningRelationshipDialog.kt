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
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.Data.Model.Model

@Composable
fun EditFinetuningRelationshipDialog(
    onDismiss: () -> Unit,
    models: List<Model>,
    dataset: List<Dataset>,
    onSubmit: (LocalRelationship) -> Unit
) {

    var selectedModel by remember { mutableStateOf("") }
    var selectedSourceIds by remember { mutableStateOf(emptyList<String>()) }
    var expandedModel by remember { mutableStateOf(false) }
    var expandedSourceIds by remember { mutableStateOf(false) }

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
                    Text(
                        text = selectedModel.ifEmpty { "Select Model" },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedModel = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary)
                    )
                    DropdownMenu(
                        expanded = expandedModel,
                        onDismissRequest = { expandedModel = false }
                    ) {
                        models.forEach { model ->
                            DropdownMenuItem(
                                text = { Text(model.uniqueIdentifier) },
                                onClick = {
                                    selectedModel = model.uniqueIdentifier
                                    expandedModel = false
                                }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text("Source Unique Identifiers")
                Box {
                    Text(
                        text = if (selectedSourceIds.isEmpty()) "Select Source Dataset" else selectedSourceIds.joinToString(),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedSourceIds = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary)
                    )
                    DropdownMenu(
                        expanded = expandedSourceIds,
                        onDismissRequest = { expandedSourceIds = false }
                    ) {
                        dataset.forEach { data ->
                            val isSelected = selectedSourceIds.contains(data.uniqueIdentifier)
                            DropdownMenuItem(
                                text = {
                                    Text(data.uniqueIdentifier + if (isSelected) " (Selected)" else "")
                                },
                                onClick = {
                                    if (isSelected) {
                                        selectedSourceIds =
                                            selectedSourceIds - data.uniqueIdentifier
                                    } else {
                                        selectedSourceIds =
                                            selectedSourceIds + data.uniqueIdentifier
                                    }
                                }
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        val localRelationship = LocalRelationship(
                            modelUniqueIdentifier = selectedModel,
                            relationshipType = "Dataset",
                            sourceUniqueIdentifiers = selectedSourceIds.joinToString(",")
                        )
                        onSubmit(localRelationship)
                    },
                    enabled = selectedModel.isNotEmpty() && selectedSourceIds.isNotEmpty(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Submit")
                }
            }
        }
    }
}