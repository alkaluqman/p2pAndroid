package com.example.applayout.Data.Model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "LocalRelationships")
data class LocalRelationship(
    @PrimaryKey val modelUniqueIdentifier: String,
    val relationshipType: String,
    val sourceUniqueIdentifiers: String //Json String
)

