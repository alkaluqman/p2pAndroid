package com.example.applayout.Models

import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.applayout.Data.Model.Model
import com.example.applayout.Marketplace.downloadModelFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun PublicModelCard(filesDir: File, modelData: Model, navController: NavController) {
    val scope = rememberCoroutineScope()
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clickable {
                val encodedUrl =
                    Uri.encode("https://android-p2p-frontend-xoxm-p3s5edewc-yu-feis-projects.vercel.app/weight/${modelData.uniqueIdentifier}")
//                    Uri.encode("http://10.0.2.2:3000/weight/${modelData.uniqueIdentifier}")
                navController.navigate("webview/$encodedUrl")
            },
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(end = 16.dp)
            ) {
                Text(
                    text = "Model: ${modelData.uniqueIdentifier}",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                Text(
                    text = "File Size: ${modelData.weight_size} bytes",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }
            IconButton(onClick = {
                scope.launch(Dispatchers.IO) {
                    downloadModelFile(modelData.public_link, filesDir, modelData.uniqueIdentifier)
                }
            }) {
                Icon(
                    imageVector = Icons.Default.FavoriteBorder,
                    contentDescription = "Download Model",
                    tint = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}