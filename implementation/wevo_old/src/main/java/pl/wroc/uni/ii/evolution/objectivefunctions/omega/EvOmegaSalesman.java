package pl.wroc.uni.ii.evolution.objectivefunctions.omega;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Function that solves Salesman Traveling Problem
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaSalesman implements EvObjectiveFunction<EvOmegaIndividual> {

  private static final long serialVersionUID = 42874242443163144L;

  private double cities[][];


  public EvOmegaSalesman(double cities[][]) {
    this.cities = cities;
  }


  public double evaluate(EvOmegaIndividual individual) {
    ArrayList<Integer> solution = individual.getFenotype();
    double sum = 0.0;
    int solution_len = solution.size();
    int city1 = solution.get(0);

    for (int i = 0; i < solution_len; i++) {
      int pos = (i + 1) % solution_len;
      int city2 = solution.get(pos);
      sum += cities[city1][city2];
      city1 = city2;
    }

    return -sum;
  }

}
