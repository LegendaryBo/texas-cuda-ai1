package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

public class EvRealVectorDifferentialEvolutionOperator implements
    EvOperator<EvRealVectorIndividual> {

  private final double delta;


  public EvRealVectorDifferentialEvolutionOperator(double delta) {
    this.delta = delta;
  }


  public EvPopulation<EvRealVectorIndividual> apply(
      EvPopulation<EvRealVectorIndividual> population) {
    EvPopulation<EvRealVectorIndividual> result =
        new EvPopulation<EvRealVectorIndividual>();
    while (result.size() < population.size()) {
      EvRealVectorIndividual r1 =
          population.get(EvRandomizer.INSTANCE.nextInt(population.size()))
              .clone();
      EvRealVectorIndividual r2 =
          population.get(EvRandomizer.INSTANCE.nextInt(population.size()));
      EvRealVectorIndividual r3 =
          population.get(EvRandomizer.INSTANCE.nextInt(population.size()));
      for (int i = 0; i < r1.getDimension(); i++) {
        r1.setValue(i, r1.getValue(i) + delta * (r2.getValue(i))
            - r3.getValue(i));
      }
      result.add(r1);
    }
    return result;
  }

}
