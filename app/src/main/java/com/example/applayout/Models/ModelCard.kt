package com.example.applayout.Models

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.applayout.Data.Model.Model

private const val BASE_URL = "10.96.181.80"
@Composable
fun ModelCard(
    isSelected: Boolean,
    modelData: Model,
    navController: NavController,
    onClick: (String) -> Unit,
    onEdit: (Model) -> Unit,
    onDelete: (filename: String) -> Unit,
    onUpload: (String) -> Unit,
    isUploaded: Boolean
) {
    val backgroundColor =
        if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.1f) else MaterialTheme.colorScheme.surface
    val borderColor =
        if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(
            alpha = 0.2f
        )
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .background(backgroundColor)
            .border(2.dp, borderColor)
            .clickable {
                onClick(modelData.uniqueIdentifier)
            },
        elevation = CardDefaults.cardElevation(4.dp)
    ) {

        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = modelData.uniqueIdentifier,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(end = 16.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(2f)) {
                            InfoText("Architecture:", modelData.architecture)
                            InfoText("Model Task:", modelData.model_task)
                            InfoText("File Size:", "${modelData.weight_size} bytes")
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            InfoText("Usage:", modelData.usage.toString())
                            InfoText("Likes:", modelData.likes.toString())
                        }
                    }
                }
                Row {
                    IconButton(
                        enabled = modelData.isOwner,
                        onClick = {
                            onEdit(modelData)
                        }) {
                        Icon(
                            imageVector = Icons.Default.Create,
                            contentDescription = "Edit Model",
                        )
                    }
                    if (!isUploaded) {
                        IconButton(onClick = { onUpload(modelData.uniqueIdentifier) }) {
                            Icon(
                                imageVector = Icons.Default.Send,
                                contentDescription = "Upload Model",
                            )
                        }
                    }
                    IconButton(onClick = {
                        val encodedUrl =
                            Uri.encode("https://android-p2p-frontend-xoxm.vercel.app/weight/${modelData.uniqueIdentifier}")
//                    Uri.encode("http://$BASE_URL:3000/weight/${modelData.uniqueIdentifier}")
                        navController.navigate("webview/$encodedUrl")
                    }) {
                        Icon(
                            imageVector = Icons.Default.ExitToApp,
                            contentDescription = "Explore Model",
                        )
                    }

                    IconButton(onClick = {
                        onDelete(modelData.uniqueIdentifier)
                    }) {
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
