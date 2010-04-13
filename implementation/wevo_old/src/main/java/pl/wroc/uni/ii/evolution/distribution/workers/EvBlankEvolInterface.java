package pl.wroc.uni.ii.evolution.distribution.workers;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Blank interface which does nothing. Use it to avoid nullpointerexceptions
 * 
 * @author Kacper Gorski
 */
public class EvBlankEvolInterface implements EvEvolutionInterface {

  public void addEvalTime(int time) {
  }


  public void addExportedPopulation(List population) {
  }


  public void addImportedPopulation(List population) {
  }


  public void addOperatortime(EvOperator operator, int time) {
  }


  public void currentNode(int node_id) {
  }


  public void currentPopulation(EvPopulation population) {
  }


  public void currentTask(int task_id) {
  }


  public void iterationProgress() {
  }


  public void populationSize(int size) {
  }


  public void newTask() {
  }


  public void currentCellID(long cell_id) {
  }

}
