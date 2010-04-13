package pl.wroc.uni.ii.evolution.sampleimplementation;

import javax.swing.JEditorPane;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

public class EvSGAAppletUpdateProblemDisplay implements
    EvOperator<EvBinaryVectorIndividual> {

  JEditorPane pane;

  long update_time = 0;


  public EvSGAAppletUpdateProblemDisplay(JEditorPane problem) {
    pane = problem;
    update_time = System.currentTimeMillis();
  }


  @SuppressWarnings("unchecked")
  public EvPopulation apply(EvPopulation<EvBinaryVectorIndividual> population) {
    if (System.currentTimeMillis() - update_time < 1000) {
      return population;
    }
    int max_ind = 0;
    double max_obj = Double.MIN_VALUE;
    int s = 0;
    for (EvBinaryVectorIndividual i : population) {
      if (i.getObjectiveFunctionValue() > max_obj) {
        max_obj = i.getObjectiveFunctionValue();
        max_ind = s;
      }
      s++;
    }

    EvSGAAppletObjectiveFunction obj =
        (EvSGAAppletObjectiveFunction) population.get(max_ind)
            .getObjectiveFunction();
    pane.setText(obj.toString(population.get(max_ind)));
    update_time = System.currentTimeMillis();
    return population;
  }

}
