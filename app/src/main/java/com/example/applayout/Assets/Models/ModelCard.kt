package com.example.applayout.Assets.Models

import android.content.Intent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.applayout.Assets.getFileSize
import com.example.applayout.Database.AppDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun ModelCard(filesDir: File, modelId: String) {
    val context = LocalContext.current
    val database = AppDatabase.getDatabase(context.applicationContext)
    val modelData by database.modelDao().getModel(modelId).observeAsState()
    if (modelData != null) {
        val modelSize = remember { getFileSize(filesDir, modelData!!.filePath) }
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
                .clickable {
                    val intent = Intent(context, ModelDetailsActivity::class.java).apply {
                        putExtra("modelId", modelId)
                    }
                    context.startActivity(intent)
                },
            elevation = CardDefaults.cardElevation(4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Model: $modelId",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Model name: ${modelData!!.name}",
                    fontSize = 16.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "File Size: $modelSize bytes",
                    fontSize = 16.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                IconButton(onClick = {
                    CoroutineScope(Dispatchers.IO).launch {
                        database.modelDao().deleteModel(modelId)
                    }
                }) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete Model",
                        tint = Color.Red
                    )
                }
            }

        }

    }
}

