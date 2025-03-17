package com.example.applayout.Marketplace

import androidx.compose.runtime.*
import com.example.applayout.utils.*

@Composable
fun LastFinetuneScreen() {

    val fileName = "last_run_losses.csv"
    val xAxisTitle = "epoch"
    val yAxisTitle = "loss"

    Graph(
        fileName = fileName,
        xAxisTitle = xAxisTitle,
        yAxisTitle = yAxisTitle,
        useCase = "finetune"
    )

}


