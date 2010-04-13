package pl.wroc.uni.ii.evolution.servlets.managment;

import java.io.IOException;
import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.*;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * @author Marcin Golebiowski
 */
public class EvTasksManager {
  private ArrayList<EvTaskInfo> task_list = new ArrayList<EvTaskInfo>();

  private EvDBServletCommunication database_servlets;

  private int current_selected_task = 0;


  public EvTasksManager(EvDBServletCommunication gateway) {
    this.database_servlets = gateway;
    init();

  }


  public void init() {
    try {
      task_list = new ArrayList<EvTaskInfo>();
      Integer[] ids = database_servlets.getTaskIDsForSystem();
      if (ids == null) {

        return;
      }
      for (Integer id : ids) {
        EvTaskInfo info = database_servlets.getTaskForSystem(id);
        task_list.add(info);
      }

    } catch (IOException ex) {
      return;
    }
  }


  /**
   * Add task to task list
   * 
   * @param new_task
   */
  public void addTask(EvTaskInfo new_task) {
    task_list.add(new_task);
  }


  /**
   * Assigns task for node. It returns a task identifier.
   * 
   * @return
   */
  public int assignTask() {

    if (task_list.size() == 0) {
      return -1;
    }

    for (int i = 0; i < task_list.size(); i++) {
      current_selected_task = (current_selected_task + 1) % task_list.size();
      if (task_list.get(current_selected_task).getStatus() == 1) {
        return task_list.get(current_selected_task).getId();
      }
    }
    return -1;
  }


  /**
   * Returns list of EvTaskInfo
   * 
   * @return
   */
  public ArrayList<EvTaskInfo> getList() {
    return task_list;
  }


  public EvTaskInfo getTask(int id) {
    for (int i = 0; i < task_list.size(); i++) {
      if (task_list.get(i).getId() == id) {
        return task_list.get(i);
      }
    }
    return null;
  }


  public void deleteTask(Integer id) {
    for (int i = 0; i < task_list.size(); i++) {
      if (task_list.get(i).getId() == id) {
        task_list.remove(i);
      }
    }
  }


  public void changeTaskState(Integer id, int state) {
    for (int i = 0; i < task_list.size(); i++) {
      if (task_list.get(i).getId() == id) {
        task_list.get(i).setStatus(state);
      }
    }
  }


  public EvTaskInfo getTask(Integer id) {
    for (int i = 0; i < task_list.size(); i++) {
      if (task_list.get(i).getId() == id) {
        return task_list.get(i);
      }
    }
    return null;
  }

}
