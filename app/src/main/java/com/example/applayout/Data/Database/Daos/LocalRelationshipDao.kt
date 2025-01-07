package com.example.applayout.Data.Database.Daos


import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.applayout.Data.Model.LocalRelationship


@Dao
interface LocalRelationshipDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(relationship: LocalRelationship)

    @Query("SELECT * FROM LocalRelationships WHERE modelUniqueIdentifier = :id")
    suspend fun getById(id: String): LocalRelationship?

    @Query("DELETE FROM LocalRelationships WHERE modelUniqueIdentifier = :id")
    suspend fun deleteById(id: String)

    @Query("SELECT * FROM LocalRelationships")
    suspend fun getAll(): List<LocalRelationship>
}
