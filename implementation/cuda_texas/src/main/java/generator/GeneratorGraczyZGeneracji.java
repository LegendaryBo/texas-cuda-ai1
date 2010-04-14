package generator;

import java.util.ArrayList;
import java.util.Random;

import engine.TexasSettings;

import operatory.WywalDuplikaty;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.likeness.EvHammingDistanceLikenes;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvRestrictedTournamentReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorMultiPointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import wevo.TexasObjectiveFunction;
import wevo.statystyki.StatystykiTools;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;
import Gracze.gracz_v3.GeneratorRegulv3;

public class GeneratorGraczyZGeneracji extends IndividualGenerator {

	public static String SCIEZKA = null;
	
	public ArrayList<EvBinaryVectorIndividual> lista = new ArrayList<EvBinaryVectorIndividual>();

	private int generacja;

	boolean uzyjPoprzednichGeneracji;

	private IndividualGenerator generatorSlabych;
	private Random generatorLiczb;

	private int seed = 0;

	public GeneratorGraczyZGeneracji(int seed, int size_, int aGeneracja,
			boolean aUzyjPoprzednichGeneracji) {
		super(seed, size_);
		uzyjPoprzednichGeneracji = aUzyjPoprzednichGeneracji;
		generacja = aGeneracja;
		generatorSlabych = new IndividualGenerator(seed, size_);
		generatorLiczb = new Random(seed);
		odczytajIndividuale();
		this.seed = seed;
	}

	public EvBinaryVectorIndividual generate() {
		if (generacja == 0) {
			return generatorSlabych.generate();
		} else {
			return lista.get(generatorLiczb.nextInt(lista.size()));
		}
	}

	private void odczytajIndividuale() {

		if (generacja > 0) {
			if (uzyjPoprzednichGeneracji) {
				
				if (SCIEZKA==null) {
					SCIEZKA = TexasSettings.class.getClassLoader().getResource(
							"texas_individuale").getPath();
				}
				
				lista = new ArrayList<EvBinaryVectorIndividual>();
				for (int i = 1; i <= generacja; i++) {
//					System.out.println("odczytuje plik "
//							+ SCIEZKA + "generacja" + i
//							+ ".dat");
					lista
							.addAll(IndividualIO
									.odczytajZPliku(SCIEZKA + "/generacja"
											+ i + ".dat"));
				}
			} else {
				lista = IndividualIO
						.odczytajZPliku(SCIEZKA+"generacja"
								+ generacja + ".dat");
			}

		}

	}

	public void reset() {

		generatorLiczb = new Random(seed);

	}

	public static void GenerujGeneracje(int generacja) { // generuj
										// generacje?!

		EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();

		GeneratorRegul.init();

		System.out.println("GENERUJE GENERACE " + (generacja + 1) + ":");
		TexasObjectiveFunction objective_function = new TexasObjectiveFunction(
				2000, generacja, true, true);
		int populacja = 100;
		int genes = GeneratorRegulv3.rozmiarGenomu;
		System.out.println("rozmiar osobnika " + genes);

		int iteracji = 400;
		final int ILE_DO_PLIKU = 10;

		EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(
				populacja);

//		EvSolutionSpace solutionSpace = new TaxasSolutionSpace(
//				objective_function, 1, generacja);
		 EvSolutionSpace solutionSpace = new
		 EvBinaryVectorSpace(objective_function, genes);

		genericEA.setSolutionSpace(solutionSpace);
		genericEA.setObjectiveFunction(objective_function);

		EvRestrictedTournamentReplacement replacement = new EvRestrictedTournamentReplacement(
				30, new EvHammingDistanceLikenes());
		// EvBestFromUnionReplacement replacement2 = new
		// EvBestFromUnionReplacement();
		// EvEliteReplacement elite = new EvEliteReplacement(populacja, 0);

		EvKnaryVectorMultiPointCrossover cross = new EvKnaryVectorMultiPointCrossover(
				3);
		EvBinaryVectorNegationMutation mut = new EvBinaryVectorNegationMutation(
				0.02);
		mut.setMutateClone(true);

		genericEA.addOperatorToEnd(new EvReplacementComposition(mut,
				replacement));
		genericEA.addOperatorToEnd(new EvReplacementComposition(cross,
				replacement));

		StatystykiTools.przygotujAlgorytm(storage, genericEA);

		genericEA
				.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
						iteracji));

		genericEA.init();

		ArrayList<EvBinaryVectorIndividual> lista = new ArrayList<EvBinaryVectorIndividual>();

		for (int i = 0; i < iteracji; i++) {

			genericEA.doIteration();
			System.out.println("oblizcen funkcji celu"
					+ objective_function.licznik);
			String[] partie = objective_function.getPrzykladowePartie(1,
					genericEA.getBestResult());
			for (int j = 0; j < partie.length; j++)
				System.out.println(partie[j]);

			EvPopulation<EvBinaryVectorIndividual> pop = genericEA
					.getPopulation().kBest(ILE_DO_PLIKU);
			lista.addAll(pop);
		}

		System.out.println("bylo: " + lista.size());
		lista = WywalDuplikaty.selekcjaDoPliku(lista);
		System.out.println("jest: " + lista.size());

		for (EvBinaryVectorIndividual evBinaryVectorIndividual : lista) {
			System.out.println(evBinaryVectorIndividual
					.getObjectiveFunctionValue());
		}
		StatystykiTools.wypiszStatystyki(storage, generacja);

		IndividualIO.zapiszDoPliku(lista, SCIEZKA + "generacja"
				+ (generacja + 1) + ".dat");

	}

	public static void main(String[] args) {
		GeneratorRegulv3.init();
		
		for (int i = 9; i <= 10; i++)
			GenerujGeneracje(i);
	}

}
