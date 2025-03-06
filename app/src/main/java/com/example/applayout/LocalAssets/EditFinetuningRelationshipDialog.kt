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

    var selectedModelId by remember { mutableStateOf("") }
    var selectedDatasetId by remember { mutableStateOf("") }
    var expandedModelId by remember { mutableStateOf(false) }
    var expandedDatasetId by remember { mutableStateOf(false) }

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
                    Text(text = selectedModelId.ifEmpty { "Select Model" },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedModelId = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary))
                    DropdownMenu(expanded = expandedModelId,
                        onDismissRequest = { expandedModelId = false }) {
                        models.forEach { model ->
                            DropdownMenuItem(text = { Text(model.uniqueIdentifier) }, onClick = {
                                selectedModelId = model.uniqueIdentifier
                                expandedModelId = false
                            })
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text("Source Unique Identifiers")
                Box {
                    Text(text = selectedDatasetId.ifEmpty { "Select Source Dataset" },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expandedDatasetId = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary))
                    DropdownMenu(expanded = expandedDatasetId,
                        onDismissRequest = { expandedDatasetId = false }) {
                        dataset.forEach { data ->
                            DropdownMenuItem(text = {
                                Text(data.uniqueIdentifier)
                            }, onClick = {
                                selectedDatasetId = data.uniqueIdentifier
                                expandedDatasetId = false
                            })
                        }
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        val localRelationship = LocalRelationship(
                            modelUniqueIdentifier = selectedModelId,
                            relationshipType = "Dataset",
                            sourceUniqueIdentifiers = selectedDatasetId
                        )
                        onSubmit(localRelationship)
                    },
                    enabled = selectedModelId.isNotEmpty() && selectedDatasetId.isNotEmpty(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Submit")
                }
            }
        }
    }
}