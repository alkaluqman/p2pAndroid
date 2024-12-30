package com.example.applayout.Models

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun LocalModelCard(
    fileName: String,
    fileSize: Long,
    onEdit: (String) -> Unit,
    onSend: (String) -> Unit,
    onDelete: (filename: String) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = fileName,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "File Size: $fileSize bytes",
                    fontSize = 16.sp
                )
                Row() {
                    IconButton(onClick = { onEdit(fileName) }) {
                        Icon(
                            imageVector = Icons.Default.Create,
                            contentDescription = "Edit Model",
                        )
                    }
                    IconButton(onClick = { onSend(fileName) }) {
                        Icon(
                            imageVector = Icons.Default.Send,
                            contentDescription = "Upload Model",
                        )
                    }
                    IconButton(onClick = { onDelete(fileName) }) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Delete Model",
                        )
                    }
                }

            }
        }
    }
}