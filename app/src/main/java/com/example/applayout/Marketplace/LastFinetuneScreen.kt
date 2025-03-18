package com.example.applayout.Marketplace

import android.content.Context
import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.applayout.Data.Model.Finetune
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.utils.*
import com.example.applayout.Models.readFloatsFromFile

@Composable
fun LastFinetuneScreen(context: Context, localRelationship: LocalRelationship, finetune: Finetune) {

    val tag = "LastFinetuneScreen"

    val fileName = "last_run_losses.csv"
    val xAxisTitle = "epoch"
    val yAxisTitle = "loss"

    val evalLosses = readFloatsFromFile(context = context, fileName = "last_run_eval_losses")
    Log.d(tag, "evalLosses: $evalLosses")
    val evalEpochs = List(evalLosses.size) { it.toFloat() }


    Log.d(tag, "localRelationship: $localRelationship")
    Log.d(tag, "finetune: $finetune")

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp) // Add more space if needed
    ) {
        Graph(
            fileName = fileName,
            xAxisTitle = xAxisTitle,
            yAxisTitle = yAxisTitle,
            useCase = "finetune",
        )

        Graph(
            xAxisTitle = "dataset images",
            yAxisTitle = yAxisTitle,
            xAxisData = evalEpochs,
            yAxisData = evalLosses,
            useCase = "model evaluation"
        )
    }

}
