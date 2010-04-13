package pl.wroc.uni.ii.evolution.servlets.databasesupport.structure;

import java.sql.Timestamp;

/**
 * Object of this class contains individual and all information about individual
 * 
 * @author Piotr Lipinski, Marcin Golebiowski
 */
public class EvIndividualInfo {

  private int id;

  private long task_id;

  private double objective_value;

  private Object individual;

  private long cell_id;

  private long node_id;

  private Timestamp creationTime;


  public int getID() {
    return id;
  }


  public void setID(int ID) {
    this.id = ID;
  }


  public long getTaskID() {
    return task_id;
  }


  public void setTaskID(long instanceID) {
    this.task_id = instanceID;
  }


  public double getObjectiveValue() {
    return objective_value;
  }


  public void setObjectiveValue(double objectiveValue) {
    this.objective_value = objectiveValue;
  }


  public Object getIndividual() {
    return individual;
  }


  public void setIndividual(Object individual) {
    this.individual = individual;
  }


  public long getCellID() {
    return cell_id;
  }


  public void setCellID(long creationCell) {
    this.cell_id = creationCell;
  }


  public long getNodeID() {
    return node_id;
  }


  public void setNodeID(long creationNode) {
    this.node_id = creationNode;
  }


  public Timestamp getCreationTime() {
    return creationTime;
  }


  public void setCreationTime(Timestamp creationTime) {
    this.creationTime = creationTime;
  }
}
