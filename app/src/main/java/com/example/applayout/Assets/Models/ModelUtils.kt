package com.example.applayout.Assets.Models

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.applayout.Database.Entities.Model
import java.util.UUID

@Composable
fun AddModelDialog(
    onDismiss: () -> Unit,
    onAddModel: (Model) -> Unit
) {
    var modelName by remember { mutableStateOf("") }
    var modelDescription by remember { mutableStateOf("") }
    var modelTask by remember { mutableStateOf("") }
    var modelArchitecture by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = {
                if (modelName.isNotBlank()) {
                    val modelData = Model(
                        modelId = UUID.randomUUID().toString(),
                        name = modelName,
                        description = modelDescription,
                        task = modelTask,
                        architecture = modelArchitecture
                    )
                    onAddModel(modelData)
                    onDismiss()
                }
            }) {
                Text("Add")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
        title = { Text("Add New Model") },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Model name:")
                BasicTextField(
                    value = modelName,
                    onValueChange = { modelName = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("Description:")
                BasicTextField(
                    value = modelDescription,
                    onValueChange = { modelDescription = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("Task:")
                BasicTextField(
                    value = modelTask,
                    onValueChange = { modelTask = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))

                Text("Architecture:")
                BasicTextField(
                    value = modelArchitecture,
                    onValueChange = { modelArchitecture = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )
            }
        }
    )
}