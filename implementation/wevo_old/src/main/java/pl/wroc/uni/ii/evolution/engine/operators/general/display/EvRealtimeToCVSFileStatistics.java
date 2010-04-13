package pl.wroc.uni.ii.evolution.engine.operators.general.display;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * @author Donata Malecka, Piotr Baraniak
 */

public class EvRealtimeToCVSFileStatistics<T extends EvIndividual> implements
    EvOperator<T> {

  BufferedWriter file_out;


  public EvRealtimeToCVSFileStatistics(String file_name) {
    try {
      file_out = new BufferedWriter(new FileWriter(file_name));
    } catch (IOException e) {
      System.out.println("There was problem with file");
    }
  }


  @Override
  protected void finalize() throws Throwable {
    file_out.close();
    super.finalize();
  }


  public EvPopulation<T> apply(EvPopulation<T> population) {

    double sum = 0.0;
    double variance = 0;

    for (int i = 0; i < population.size(); i++) {
      sum += population.get(i).getObjectiveFunctionValue();
    }
    double mean = sum / (double) population.size();

    for (int i = 0; i < population.size(); i++) {

      variance +=
          Math.pow((population.get(i).getObjectiveFunctionValue() - mean), 2);

    }
    variance = variance / (double) population.size();

    try {
      // file_out.write(population.getBestResult());
      // file_out.write(population.getWorstResult());
      file_out.write(population.getBestResult().getObjectiveFunctionValue()
          + ";");
      file_out.write(mean + ";");
      file_out.write(population.getWorstResult().getObjectiveFunctionValue()
          + ";");
      file_out.write(variance + "\n");
      file_out.flush();

    } catch (IOException e) {
      System.out.println("problem with writing into file");
    }
    return population;
  }
}
