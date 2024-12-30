package com.example.applayout.LocalAssets

import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
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
import com.example.applayout.Data.Model.Model
import com.example.applayout.Data.Model.UploadFormData

@Composable
fun EditModelDialog(onDismiss: () -> Unit, modelData: Model, onSubmit: (UploadFormData) -> Unit) {
    var modelTask by remember { mutableStateOf(modelData.model_task) }
    var description by remember { mutableStateOf(modelData.description) }
    var architecture by remember { mutableStateOf(modelData.architecture) }
    val modelTasks =
        listOf("object_detection", "image_classification", "segmentation", "regression")

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(dismissOnBackPress = true, dismissOnClickOutside = true)
    ) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.padding(16.dp),
            tonalElevation = 4.dp
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    "Upload Weight",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                Text("Model Task")
                var expanded by remember { mutableStateOf(false) }
                Box {
                    Text(
                        text = modelTask,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expanded = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary)
                    )
                    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        modelTasks.forEach { task ->
                            DropdownMenuItem(
                                text = { Text(task) },
                                onClick = {
                                    modelTask = task
                                    expanded = false
                                }
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text("Description")
                BasicTextField(
                    value = description,
                    onValueChange = { description = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.dp, MaterialTheme.colorScheme.primary)
                        .padding(8.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("Architecture")
                BasicTextField(
                    value = architecture,
                    onValueChange = { architecture = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.dp, MaterialTheme.colorScheme.primary)
                        .padding(8.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        onSubmit(
                            UploadFormData(
                                model_task = modelTask,
                                description = description,
                                architecture = architecture
                            )
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Submit")
                }
            }
        }
    }
}