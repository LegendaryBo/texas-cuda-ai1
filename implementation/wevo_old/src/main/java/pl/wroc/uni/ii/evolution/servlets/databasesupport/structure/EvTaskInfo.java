package pl.wroc.uni.ii.evolution.servlets.databasesupport.structure;

import java.io.Serializable;
import java.sql.Timestamp;

/**
 * Object of this class contains all basic information about task for system.
 * 
 * @author Marcin Golebiowski
 */
public class EvTaskInfo implements Serializable {

  private static final long serialVersionUID = -8527982704961398146L;

  private int id;

  private Object jar;

  private int status;

  private Timestamp submission_time;

  private String description;


  /**
   * Returns description of task
   * 
   * @return String
   */
  public String getDescription() {
    return description;
  }


  /**
   * Sets description of task
   * 
   * @param description short description (max. 255 chars)
   */
  public void setDescription(String description) {
    this.description = description;
  }


  /**
   * Returns identifier of task
   * 
   * @return int
   */
  public int getId() {
    return id;
  }


  /**
   * Sets identifier for task
   * 
   * @param id
   */
  public void setId(int id) {
    this.id = id;
  }


  public Object getJar() {
    return jar;
  }


  public void setJar(Object jar) {
    this.jar = jar;
  }


  /**
   * Returns status of task. <br />
   * Possible values:
   * <ul>
   * <li> 1 - task submited
   * <li> 2 - task stopped
   * </ul>
   * 
   * @return int
   */
  public int getStatus() {
    return status;
  }


  /**
   * Sets status of task
   * 
   * @param status
   */
  public void setStatus(int status) {
    this.status = status;
  }


  /**
   * Returns when task was submitted
   * 
   * @return Timestamp
   */
  public Timestamp getSubmissionTime() {
    return submission_time;
  }


  /**
   * Sets when task was submitted
   * 
   * @param submission_time
   */
  public void setSubmissionTime(Timestamp submission_time) {
    this.submission_time = submission_time;
  }
}