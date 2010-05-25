package wevo;

import engine.Gra;
import engine.rezultaty.Rezultat;
import generator.GeneratorGraczyZGeneracji;
import generator.IndividualGenerator;
import generator.ProstyGeneratorLiczb;
import generator.SimpleIndividualGenerator;

import java.util.Random;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import Gracze.Gracz;
import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;

/**
 * 
 * Funkcja celu
 * 
 * Wartoscia funkcji celu jest sumaryczna wartosc przegranej/wygranej
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 * 
 */
public class TexasObjectiveFunction implements
		EvObjectiveFunction<EvBinaryVectorIndividual> {

	private static final long serialVersionUID = 7051183109319973348L;

	// ilosc gier z jednym zestawem graczy
	// zwykle kilka K -10 K
	private int games = 50;

	// seed
	Random generator_gier = null;

	// takie podreczne seedy
	int seed = 61223233;
	int seed2 = 11231232;
	int seed3 = 6121365;

	Random generator_kolejnosci = null;

	// jak dobrzy osobnicy graja
	int trudnosc = 0;

	IndividualGenerator generator_individuali = null;

	private boolean uzyjPoprzednichGeneracji = false;

	private boolean resetRandomow;

	// tylko do testow porownawczych!
	public TexasObjectiveFunction(int games_) {
		this(games_, -1, false, false);

		final int LICZBA_GENOW=GeneratorRegulv3.rozmiarGenomu;
		final int LICZBA_OSOBNIKOW=100;
		GeneratorGraczyZGeneracji generator = new GeneratorGraczyZGeneracji(1234, LICZBA_GENOW, 11, true);
		EvBinaryVectorIndividual[] osobniki_java = new EvBinaryVectorIndividual[LICZBA_OSOBNIKOW];
		ProstyGeneratorLiczb random = new ProstyGeneratorLiczb(465);
		int[][] osobniki = new int[LICZBA_OSOBNIKOW][];
		for (int i=0; i < LICZBA_OSOBNIKOW; i++) {
			int losowaLiczba = random.nextInt( generator.lista.size() );
			osobniki_java[i] = generator.lista.get( losowaLiczba );
		}
		
		generator_individuali = new SimpleIndividualGenerator(games_, LICZBA_GENOW, osobniki_java);
	}
	
	public TexasObjectiveFunction(int games_, int dlugosc_osobnika,
			EvBinaryVectorIndividual[] partnerzy) {
		this(games_, -1, false, false);

		generator_individuali = new SimpleIndividualGenerator(games_, dlugosc_osobnika, partnerzy);
	}

	/**
	 * Tworzy funkcje celu, ktora rozgrywa 50 gier z podanym zestawem
	 * osobnikow o podanej trudnosci
	 * 
	 * 
	 * 
	 * @param games_
	 * @param trudnosc_
	 *                : 0 - osobnicy zupelnie losowi 1 - osobnicy sredni,
	 *                wytrenowani na losowych i zapisani do pliku 2 -
	 *                osobnicy dobrzy, wytrenowani na srednich
	 */
	public TexasObjectiveFunction(int games_, int trudnosc_,
			boolean aUzyjPoprzednichGeneracji, boolean resetRandomow) {
		this.resetRandomow = resetRandomow;
		games = games_;
		trudnosc = trudnosc_;

		uzyjPoprzednichGeneracji = aUzyjPoprzednichGeneracji;

		generator_kolejnosci = new Random(43435);
		generator_gier = new Random(43435);
		seed3 = 43435;

	}

	public TexasObjectiveFunction(int games_, int trudnosc_,
			boolean aUzyjPoprzednichGeneracji, int aSeed,
			boolean resetRandomow) {

		this(games_, trudnosc_, aUzyjPoprzednichGeneracji,
				resetRandomow);
		this.resetRandomow = resetRandomow;

		Random pGenSeedow = new Random(aSeed);

		generator_kolejnosci = new Random(pGenSeedow.nextInt());
		generator_gier = new Random(pGenSeedow.nextInt());
		// seed3 = pGenSeedow.nextInt();
	}

	public static int licznik = 0;

	public static boolean LOGI=false;
	
	/**
	 * po prostu rozgrywa podana liczbe parti i zwraca sumaryczny bilans
	 */
	public double evaluate(EvBinaryVectorIndividual individual) {

		licznik++;
		// System.out.println(iter);

		if (generator_individuali == null) {

			if (trudnosc == 0)
				generator_individuali = new IndividualGenerator(
						seed3, individual
								.getDimension());
			if (trudnosc > 0)
				generator_individuali = new GeneratorGraczyZGeneracji(
						seed3,
						individual.getDimension(),
						trudnosc,
						uzyjPoprzednichGeneracji);
		}

		if (resetRandomow) {

			generator_kolejnosci = new Random();
			generator_gier = new Random();

			if (trudnosc > 0)
				generator_individuali.reset();

		} else {
			generator_kolejnosci = new Random(43435);
			generator_gier = new Random(43435);
		}

		double bilans = 0;
		double wygrana = 0;
		double przegrana = 0;

		int wygranych = 0;
		int przegranych = 0;
		int passow = 0;

		int[] licznikPassow = new int[4];
		individual.kartaWygranego = new int[7];

		individual.wygranaStawka = new int[8];
		int[] licznikWygranych = new int[8];

		// dla kazdej gry dobiera losowych 6 graczy
		for (int i = 0; i < games; i++) {

			Gracz[] gracze = new Gracz[6];

			for (int j = 0; j < 6; j++)
				gracze[(j +i)%6] = new GraczAIv3(generator_individuali
						.generate(), j);

			int random = generator_kolejnosci.nextInt(6);
			if (trudnosc<0)
				random=i%6;
			GraczAIv3 gracz = new GraczAIv3(individual, random);
			gracze[random] = gracz;

			int nr_rozdania = generator_gier.nextInt();
			if (trudnosc<0)
				nr_rozdania = i;
				
			Gra gra = new Gra(gracze, nr_rozdania);
			gra.play_round(false);

			bilans += gracz.bilans;
//			boolean[] spasowanie = new boolean[6];
//			int[] wygraniii = Gra.sprawdzenie_kart(spasowanie, gra.rozdanie);
//			for (int j=0; j < wygraniii.length; j++) {
//				bilans += wygraniii[j]*wygraniii[j];
//			}
//			bilans += gra.rozdanie.g
			if (LOGI)
			System.out.println("bilans java po "+(i+1)+" grach:"+bilans/games);
//			
//			for (int j=1; j < 6; j++)
//				System.out.print( " hash"+(j+1)+"="+((GraczAIv3)gracze[j]).individual.getJakisHashcode() );
//			
//			System.out.println();
			
			if (gracz.bilans < 0 && !gra.pass[random]) {
				przegranych++;
				przegrana -= gracz.bilans;
			}
			if (gracz.bilans > 0) {
				wygranych++;
				wygrana += gracz.bilans;

				int karta = gra.kartaWygranego[random];

				if (karta != 0) {

					if (karta > 6) {
						karta = 7;
					}
					licznikWygranych[karta - 1]++;
					individual.wygranaStawka[karta - 1] += gracz.bilans;

					individual.kartaWygranego[karta - 1]++;
				}

			}
			if (gra.pass[random]) {
				passow++;
				licznikPassow[gra.pass_runda[random]]++;
			}

			/*
			 * double bil = 0; for (int j = 0; j < 6; j++) bil +=
			 * gracze[j].bilans; System.out.println(bil);
			 * 
			 * 
			 * 
			 * for (int j = 0; j < 6; j++)
			 * System.out.println(gracze[j] + " "
			 * +RegulyGry.najlepsza_karta
			 * (gra.rozdanie.getAllCards(j)));
			 * 
			 * System.out.println(gra.rozdanie);
			 */

		}

		for (int i = 0; i < 8; i++)
			if (licznikWygranych[i] != 0)
				individual.wygranaStawka[i] /= licznikWygranych[i];

		float[] statystyki = new float[5];
		statystyki[0] = przegranych;
		statystyki[1] = wygranych;
		statystyki[2] = passow;
		if (wygranych == 0)
			statystyki[3] = 0.0f;
		else
			statystyki[3] = (float) (wygrana / wygranych);

		if (przegranych == 0)
			statystyki[4] = 0.0f;
		else
			statystyki[4] = (float) (przegrana / przegranych);

		individual.statystyki = statystyki;
		individual.licznikPassow = licznikPassow;

//		if (passow == games)
//			return -20000.0d;
//		else
			return bilans / games;
	}

	public String[] getPrzykladowePartie(int games,
			EvBinaryVectorIndividual individual) {

		if (resetRandomow) {

			generator_kolejnosci = new Random();
			generator_gier = new Random();

			if (trudnosc > 0)
				generator_individuali.reset();

		} else {
			generator_kolejnosci = new Random(43435);
			generator_gier = new Random(43435);
		}

		String[] partia = new String[games];

		// dla kazdej gry dobiera losowych 6 graczy
		for (int i = 0; i < games; i++) {

			GraczAIv3[] gracze = new GraczAIv3[6];

			for (int j = 0; j < 6; j++)
				gracze[j] = new GraczAIv3(generator_individuali
						.generate(), j);

			int random = generator_kolejnosci.nextInt(6);
			GraczAIv3 gracz = new GraczAIv3(individual, random);
			gracze[random] = gracz;

			String graczeIntro = new String();
			graczeIntro += "Nasz osobnik ma nr. " + (random + 1)
					+ " \n";
			graczeIntro += "Gracz 1, wartosc funkcji celu "
					+ evaluate(gracze[0].individual) + "\n";
			graczeIntro += "Gracz 2, wartosc funkcji celu "
					+ evaluate(gracze[1].individual) + "\n";
			graczeIntro += "Gracz 3, wartosc funkcji celu "
					+ evaluate(gracze[2].individual) + "\n";
			graczeIntro += "Gracz 4, wartosc funkcji celu "
					+ evaluate(gracze[3].individual) + "\n";
			graczeIntro += "Gracz 5, wartosc funkcji celu "
					+ evaluate(gracze[4].individual) + "\n";
			graczeIntro += "Gracz 6, wartosc funkcji celu "
					+ evaluate(gracze[5].individual) + "\n";

			Gra gra = new Gra(gracze, generator_gier.nextInt());
			partia[i] = gra.play_round(true);
			partia[i] = graczeIntro += partia[i];
		}

		return partia;
	}

	// to samo co evaluate, ale zwraca string ze statystykami
	public float[] getStats(EvBinaryVectorIndividual individual) {

		if (generator_individuali == null) {
			if (trudnosc == 0)
				generator_individuali = new IndividualGenerator(
						seed3, individual
								.getDimension());
			if (trudnosc > 0)
				generator_individuali = new GeneratorGraczyZGeneracji(
						seed3,
						individual.getDimension(),
						trudnosc,
						uzyjPoprzednichGeneracji);
		}

		double bilans = 0;
		double wygrana = 0;
		double przegrana = 0;

		int wygranych = 0;
		int przegranych = 0;
		int passow = 0;

		// dla kazdej gry dobiera losowych 6 graczy
		for (int i = 0; i < games; i++) {

			Gracz[] gracze = new Gracz[6];

			for (int j = 0; j < 6; j++)
				gracze[j] = new GraczAIv3(generator_individuali
						.generate(), j);

			int random = generator_kolejnosci.nextInt(6);
			GraczAIv3 gracz = new GraczAIv3(individual, random);
			gracze[random] = gracz;

			Gra gra = new Gra(gracze, generator_gier.nextInt());
			gra.play_round(false);
			bilans += gracz.bilans;
			

			if (gracz.bilans < 0 && !gra.pass[random]) {
				przegranych++;
				przegrana -= gracz.bilans;
			}
			if (gracz.bilans > 0) {
				wygranych++;
				wygrana += gracz.bilans;
			}
			if (gra.pass[random])
				passow++;

		}

		float[] statystyki = new float[5];
		statystyki[0] = przegranych;
		statystyki[1] = wygranych;
		statystyki[2] = passow;
		statystyki[3] = (float) (wygrana / wygranych);
		statystyki[4] = (float) (przegrana / przegranych);

		return statystyki;
		// return
		// "\nPrzegranych: "+przegranych+"  \nWygranych: "+wygranych +
		// " \nPassow "+passow+" \nSrednia wygrana "
		// +(wygrana/wygranych) + " \nSrednia przegrana " +
		// (przegrana/przegranych) ;
	}

}
