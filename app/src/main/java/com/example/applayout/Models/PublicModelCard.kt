package com.example.applayout.Models

import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Favorite
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
fun PublicModelCard(
    modelData: Model, navController: NavController,
    onDownload: () -> Unit,
    onLike: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clickable {
                val encodedUrl =
                    Uri.encode("http://$BASE_URL:3000/weight/${modelData.uniqueIdentifier}")
                navController.navigate("webview/$encodedUrl")
            },
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = modelData.uniqueIdentifier,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                modifier = Modifier.padding(bottom = 4.dp),
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
                    IconButton(onClick = { onLike() }) {
                        Icon(
                            imageVector = Icons.Default.Favorite,
                            contentDescription = "Like Model",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    IconButton(onClick = {
                        onDownload()
                    }) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "Download Model",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }

            }
        }
    }
}

@Composable
fun InfoText(label: String, value: String) {
    Text(
        text = "$label $value",
        fontSize = 13.sp,
        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
    )
}