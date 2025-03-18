package com.example.applayout.Data.Model

import android.os.Parcelable
import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlinx.parcelize.Parcelize

@Entity(tableName = "LocalRelationships")
@Parcelize
data class LocalRelationship(
    @PrimaryKey val modelUniqueIdentifier: String,
    val relationshipType: String,
    val sourceUniqueIdentifiers: String //Json String
) : Parcelable

