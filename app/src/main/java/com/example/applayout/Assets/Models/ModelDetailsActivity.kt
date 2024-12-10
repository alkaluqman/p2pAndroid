package com.example.applayout.Assets.Models

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.example.applayout.Database.AppDatabase
import com.example.applayout.Database.Entities.Model
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ModelDetailsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val modelId = intent.getStringExtra("modelId")
        if (modelId == null) {
            Toast.makeText(this, "Invalid model data", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val database = AppDatabase.getDatabase(applicationContext)

        setContent {
            MaterialTheme {
                val modelData by database.modelDao().getModel(modelId).observeAsState()
                modelData?.let { model ->
                    ModelDetailsScreen(
                        modelData = model,
                        onSave = { updatedModelData ->
                            lifecycleScope.launch(Dispatchers.IO) {
                                updatedModelData.lastModified = System.currentTimeMillis()
                                database.modelDao().updateModelFields(
                                    updatedModelData.modelId,
                                    updatedModelData.name,
                                    updatedModelData.description,
                                    updatedModelData.architecture,
                                    updatedModelData.task
                                )
                            }
                            Toast.makeText(
                                this@ModelDetailsActivity,
                                "Metadata saved!",
                                Toast.LENGTH_SHORT
                            ).show()
                            finish()
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun ModelDetailsScreen(modelData: Model, onSave: (Model) -> Unit) {
    var modelName by remember { mutableStateOf(modelData.name) }
    var description by remember { mutableStateOf(modelData.description) }
    var modelTask by remember { mutableStateOf(modelData.task) }
    var architecture by remember { mutableStateOf(modelData.architecture) }

    Column(modifier = Modifier
        .padding(16.dp)
        .fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.primary)
                .padding(16.dp)
        ) {
            Text(
                text = "Edit Model Details",
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.align(Alignment.Center)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Model Name:")
        BasicTextField(
            value = modelName,
            onValueChange = { modelName = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Text("Description:")
        BasicTextField(
            value = description,
            onValueChange = { description = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Text("Model Task:")
        BasicTextField(
            value = modelTask,
            onValueChange = { modelTask = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Text("Architecture:")
        BasicTextField(
            value = architecture,
            onValueChange = { architecture = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = {
            onSave(
                Model(
                    modelId = modelData.modelId,
                    name = modelName,
                    description = description,
                    task = modelTask,
                    architecture = architecture,
                    lastModified = System.currentTimeMillis()
                )
            )
        }) {
            Text("Save Changes")
        }
    }
}
