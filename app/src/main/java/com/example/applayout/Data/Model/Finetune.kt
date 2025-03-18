package com.example.applayout.Data.Model

import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@Parcelize
data class Finetune(
    var num_epochs: Int = 0,
    var batch_size: Int = 0,
    var performance_json: String,
    var dataset: String
) : Parcelable

