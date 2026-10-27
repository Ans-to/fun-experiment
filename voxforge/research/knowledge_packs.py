"""Curated museum-grade encyclopedic packs for resilience & deep enrichment.

When live internet is available, VoxForge merges live Wikipedia/Wikidata/Commons
extractions on top of curated encyclopedic foundations; when firewalled or offline,
these dossiers ensure zero degradation in factual precision.
"""

from __future__ import annotations

from voxforge.models import KeyMetric, ResearchDossier, SubTopicDossier


def get_seven_wonders_dossier(topic_title: str) -> ResearchDossier:
    """Comprehensive historical, structural, and archaeological dossier on the 7 Wonders."""
    subtopics = [
        SubTopicDossier(
            index=1,
            title="Origins of Wonder: From Greek Theamata to Global Monuments",
            subtitle="Why humanity obsessively catalogs the impossible — and what vanished along the way",
            era_or_date="c. 2nd Century BCE – 2007 CE",
            location="Mediterranean Basin & Global Heritage Sites",
            summary=(
                "The concept of Seven Wonders began not as a bucket list of tourist attractions, "
                "but as Hellenistic travelogues compiled by writers like Antipater of Sidon and Philo of Byzantium "
                "who called them theamata — literally 'things that must be seen.'"
            ),
            origin_and_context=[
                "Hellenistic scholars chose the number seven because in classical Greek cosmology it represented completeness and perfection: five known planets plus the Sun and Moon.",
                "Six of the original Seven Ancient Wonders — the Colossus of Rhodes, the Lighthouse of Alexandria, the Mausoleum at Halicarnassus, the Temple of Artemis at Ephesus, the Statue of Zeus at Olympia, and the Hanging Gardens — were erased by earthquakes, fires, and medieval stone-quarrying.",
                "Oxford Assyriologist Stephanie Dalley demonstrated through cuneiform prisms and aqueduct ruins at Jerwan that the famed Hanging Gardens may have been built not in Babylon by Nebuchadnezzar II, but three hundred miles north at Nineveh by Assyrian king Sennacherib around 700 BCE using bronze Archimedes-style water screws.",
                "The Pharos Lighthouse of Alexandria rose roughly 103 to 118 meters above the harbor island, projecting sunbeams by day with polished bronze mirrors and furnace firelight visible nearly fifty kilometers offshore by night until fourteenth-century Mediterranean earthquakes toppled its masonry into the sea.",
                "In 2007, more than one hundred million global ballots established the New Seven Wonders canon while designating the Great Pyramid of Giza as an honorary eternal candidate — bridging four and a half millennia of human engineering.",
            ],
            engineering_and_mechanics=[
                "Every surviving monument shares three non-negotiable engineering traits: seismic resilience, mastery of local geology, and hydrological control against water erosion.",
                "Ancient bronze wonders like the thirty-three-meter Colossus of Rhodes failed rapidly because iron internal armatures corroded and fractured under lateral seismic shear in 226 BCE, only fifty-four years after sculptor Chares of Lindos completed it.",
                "By contrast, compression-dominant stone geometries — pyramids, corbeled vaults, Roman pozzolanic concrete arches, and Inca battered walls — turn gravity itself into a stabilizing clamp.",
                "Comparing ancient logistics reveals an astonishing pattern: pre-industrial civilizations routinely mobilized twenty thousand to thirty-five thousand skilled rotating laborers without wheeled cranes, iron pulleys, or fossil fuels.",
            ],
            human_stories_and_paradoxes=[
                "Antipater of Sidon wrote in 140 BCE that after seeing the walls of Babylon, the Zeus of Olympia, and the Hanging Gardens, every other marvel dimmed beside the towering sanctuary of Artemis in Ephesus — yet today only a single reconstructed column stands in a marshy Turkish field.",
                "A monument's survival depends as much on economics as physics: once an empire's religious or civic identity shifts, bronze gets melted for coinage and dressed marble gets burned into lime mortar.",
            ],
            legacy_and_modern_science=[
                "Twenty-first-century archaeology has replaced pickaxes with non-invasive physics: airborne LiDAR strips away jungle canopies, cosmic-ray muon detectors x-ray solid limestone mountains, and isotopic geochemistry traces individual stone blocks back to exact ancient quarries.",
                "Studying these monuments together exposes a universal truth of civilization: societies build at the edge of structural collapse precisely when they want to anchor collective memory against time.",
            ],
            metrics=[
                KeyMetric("Original Canon", "c. 140 BCE (Antipater)"),
                KeyMetric("Ancient Survivors", "1 of 7 (Giza)"),
                KeyMetric("Pharos Height", "~110 Meters"),
                KeyMetric("Modern Global Poll", "100M+ Votes (2007)"),
            ],
            visual_queries=[
                "Great Pyramid of Giza panorama",
                "Seven Wonders Ancient World historical illustration",
            ],
            source_urls=["https://en.wikipedia.org/wiki/Wonders_of_the_World"],
        ),
        SubTopicDossier(
            index=2,
            title="The Great Pyramid of Giza: Precision in Two Million Stones",
            subtitle="The sole surviving Ancient Wonder — bronze-age logistics, celestial alignment, and muon secrets",
            era_or_date="c. 2560 BCE (Fourth Dynasty)",
            location="Giza Plateau, Egypt",
            summary=(
                "Built for Pharaoh Khufu under royal vizier and chief architect Hemiunu, the Great Pyramid stood "
                "146.6 meters tall — remaining the tallest human-made structure on Earth for more than 3,800 years."
            ),
            origin_and_context=[
                "Constructed over roughly twenty to twenty-seven years during Egypt's Old Kingdom around 2560 BCE, Khufu's horizon monument — originally named Akhet Khufu — crowned a vast necropolis connected directly to the Nile via artificial harbor basins.",
                "In 2013, archaeologist Pierre Tallet discovered the Diary of Merer inside Red Sea caves at Wadi al-Jarf: actual 4,500-year-old papyrus logbooks written by a middle-ranking inspector whose forty-man boat crew hauled fine white Tura limestone upriver to Giza.",
                "Excavations of the Heit el-Ghurab workers' village by Mark Lehner proved the builders were not whipped slaves, as Greek historian Herodotus imagined two millennia later, but well-fed conscripted crews organized into competitive nautical phyles eating prime beef, sheep, bread, and beer.",
            ],
            engineering_and_mechanics=[
                "The monument contains approximately 2.3 million stone blocks weighing an average of 2.5 metric tons, while the King's Chamber ceiling is spanned by nine colossal pink granite beams floated eight hundred kilometers downstream from Aswan, each weighing up to eighty tons.",
                "Surveyor Flinders Petrie measured the 230.3-meter base sides and found a mean length error of less than five centimeters and cardinal alignment to true north accurate within three-sixtieths of a single degree — likely achieved by tracking circumpolar stars with a plumb-line merkhet.",
                "Above the King's Chamber sits a five-tiered stress-relieving granite truss designed so that even when earthquakes cracked lower ceiling beams — cracks visible today — the thirty-thousand-ton overburden above could not crush the burial vault.",
                "Experimental physicists discovered that slightly wetting desert sand to roughly five percent moisture cuts sliding friction of wooden sledges in half, a technique explicitly depicted in the tomb painting of Djehutihotep.",
            ],
            human_stories_and_paradoxes=[
                "Originally encased in polished white Tura limestone angled at 51 degrees 50 minutes, the pyramid gleamed across the Nile valley like a geometric sunstar until a devastating 1303 CE earthquake loosened its outer mantle, which Mamluk sultans stripped to build Cairo's mosques and fortresses.",
                "Inside the relieving chambers sealed since antiquity, quarry gangs painted red ochre graffiti naming their work crews — including 'The Friends of Khufu' and 'The Drunkards of Menkaure' — humanizing the anonymous hands behind history's most austere geometry.",
            ],
            legacy_and_modern_science=[
                "In 2017 and 2023, the international ScanPyramids consortium used cosmic-ray muon radiography, ultrasound, and endoscopic cameras to confirm both a thirty-meter 'Big Void' above the Grand Gallery and an untouched nine-meter gabled corridor behind the North Face chevron stones.",
                "Forty-five centuries after Hemiunu laid its corner sockets into leveled Giza bedrock, the Great Pyramid still conceals sealed architectural chambers whose exact structural or ritual purpose divides Egyptologists today.",
            ],
            metrics=[
                KeyMetric("Original Height", "146.6 m (481 ft)"),
                KeyMetric("Total Stone Mass", "~6.0 Million Tons"),
                KeyMetric("Cardinal Error", "0.05° from True North"),
                KeyMetric("Record Held", "3,800+ Years Tallest"),
            ],
            visual_queries=["Great Pyramid of Giza Egypt", "Grand Gallery Khufu pyramid"],
            source_urls=["https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza"],
        ),
        SubTopicDossier(
            index=3,
            title="The Great Wall of China: Continental Defense & Sticky-Rice Mortar",
            subtitle="Twenty-one thousand kilometers of mountain fortification, optical telemetry, and Ming chemistry",
            era_or_date="7th Century BCE – 1644 CE (Ming Zenith)",
            location="Northern China Frontier",
            summary=(
                "Far from a single unbroken ribbon, the Great Wall is an evolutionary military network stretching "
                "21,196 kilometers across deserts, loess plateaus, and knife-edge granite ridges."
            ),
            origin_and_context=[
                "Rival Warring States first raised stamped-earth ramparts in the seventh century BCE, which First Emperor Qin Shi Huang linked after unifying China in 221 BCE under general Meng Tian to check nomadic Xiongnu cavalry raids from the Eurasian steppe.",
                "The iconic crenellated stone-and-brick dragon clinging to vertical peaks near Beijing, Jinshanling, and Simatai dates not to Qin antiquity, but to the Ming Dynasty after a disastrous 1449 battle at Tumu where Oirat Mongols captured the Zhengtong Emperor himself.",
                "Rather than serving purely as an impassable barrier, the wall operated as an elevated military highway allowing infantry, couriers, and supply mules to move five times faster across jagged mountain crests than raiders scrambling through ravines below.",
            ],
            engineering_and_mechanics=[
                "In 2010, analytical chemist Zhang Bingjian at Zhejiang University revealed why Ming Dynasty brickwork resisted six centuries of freeze-thaw cycles and earthquakes: builders mixed slaked lime with three percent amylopectin-rich sticky rice soup, creating an organic-inorganic composite mortar tighter and stronger than pure lime.",
                "Out in the arid Gansu corridor around Dunhuang where kiln fuel and quarry stone did not exist, Han Dynasty engineers layered compacted gravel with bundles of desert tamarisk twigs and phragmites reeds — an early fiber-reinforced composite still standing two thousand years later.",
                "Every watchtower doubled as an optical-acoustic relay node: wolf-dung or nitre-sulfur smoke plumes combined with cannon blasts transmitted invasion troop counts over five hundred kilometers to imperial command within hours.",
                "Drainage spouts carved at precise intervals channeled torrential summer monsoon runoff away from wall foundations so hydrostatic pressure could never burst the masonry.",
            ],
            human_stories_and_paradoxes=[
                "Every Ming kiln brick around Shanhaiguan and Badaling was stamped with the name of the supervising prefect and master brickmaker; if an inspector's hammer blow shattered a brick or mortar failed, the entire batch was condemned under criminal penalty.",
                "Contrary to the persistent twentieth-century myth coined twelve years before Apollo 11, astronauts confirm the Great Wall is only six to seven meters wide and matches the color of surrounding rock, making it invisible to the unaided human eye from low Earth orbit.",
                "Ironically, the wall fell in 1644 not through structural breach, but human betrayal: Ming general Wu Sangui opened the strategic Shanhaiguan 'First Pass Under Heaven' gates directly to Manchu forces after rebels seized Beijing.",
            ],
            legacy_and_modern_science=[
                "China's National Cultural Heritage Administration used GPS, infrared drones, and ground-penetrating radar to map 21,196.18 kilometers of multi-dynastic walls, trenches, and 29,000 individual fortresses and beacon towers.",
                "In Gansu province today, conservation scientists cultivate native biocrusts — living mats of cyanobacteria, moss, and lichen — that naturally armor ancient rammed-earth sections against wind erosion up to eightfold better than bare soil.",
            ],
            metrics=[
                KeyMetric("Total Surveyed Network", "21,196 km"),
                KeyMetric("Ming Dynasty Core", "8,851 km"),
                KeyMetric("Mortar Secret", "Lime + Amylopectin Rice"),
                KeyMetric("Beacon Towers", "29,000+ Nodes"),
            ],
            visual_queries=["Great Wall of China Jinshanling", "Ming Great Wall watchtower mountains"],
            source_urls=["https://en.wikipedia.org/wiki/Great_Wall_of_China"],
        ),
        SubTopicDossier(
            index=4,
            title="Petra: Hydraulic Mastery Carved into Rose-Red Sandstone",
            subtitle="How desert Nabataean nomads engineered a lush metropolis inside sheer Jordanian canyons",
            era_or_date="c. 4th Century BCE – 106 CE",
            location="Ma'an Governorate, Southern Jordan",
            summary=(
                "Hidden behind the 1.2-kilometer tectonic slot gorge of the Siq, Petra was the glittering caravan "
                "capital of the Nabataean Kingdom — carved top-down into living Paleozoic sandstone."
            ),
            origin_and_context=[
                "Originally nomadic Arab traders, the Nabataeans cornered the ancient world's most lucrative luxury supply chain: frankincense and myrrh from southern Arabia, Indian spices, and Dead Sea bitumen prized by Egypt and Rome.",
                "Their secret leverage was desert monopoly: Nabataeans excavated bottle-necked subterranean plaster cisterns across the Negev and Wadi Rum, hiding their tiny surface openings under boulders only tribal guides could locate.",
                "Approaching travelers walked for forty minutes through the shadowed, eighty-meter-high Siq fissure before emerging abruptly face-to-face with Al-Khazneh — 'The Treasury' — a thirty-nine-meter Hellenistic royal mausoleum likely commissioned by King Aretas IV in the early first century CE.",
            ],
            engineering_and_mechanics=[
                "Unlike freestanding temples built from the ground up, Nabataean stonecutters possessed zero timber scaffolding in the desert; instead, they carved stairways up the cliff, leveled an upper ledge, and chiseled facades strictly from the roof finial downward with zero margin for error.",
                "Although Petra receives scarcely ten to fifteen centimeters of winter rain per year, bare sandstone cliffs shed torrents into flash floods; Nabataean hydraulic engineers tamed the lethal Siq by boring an eighty-eight-meter diversion tunnel (Al-Mudhlim) straight through a mountain and stepping retention check-dams down every wadi.",
                "Archaeologist Charles Ortloff modeled Petra's two-hundred-kilometer network of interlocking terracotta pipes and found the Nabataeans deliberately sized pipe gradients to maintain sub-critical open-channel flow — preventing air-pocket blowouts while delivering twelve million gallons of spring water daily into fountains and terraced paradeisos gardens.",
                "Just beside the Treasury, recent excavations uncovered an entire monumental swimming pool and irrigated pleasure garden complex thriving in the middle of hyper-arid sandstone desert.",
            ],
            human_stories_and_paradoxes=[
                "Local Bedouin folklore named Al-Khazneh 'The Pharaoh's Treasury,' riddling its solid sandstone summit urn with musket pockmarks in the belief gold coins were hidden inside — when in truth the entire urn is solid carved rock.",
                "After Rome annexed Nabataea in 106 CE, maritime monsoon trade down the Red Sea bypassed overland camel routes, and a catastrophic 363 CE Galilee earthquake shattered Petra's ceramic water arteries, draining the city's lifeblood overnight.",
                "In August 1812, disguised twenty-seven-year-old Swiss explorer Johann Ludwig Burckhardt — fluent in Arabic and traveling as Sheikh Ibrahim — persuaded a local guide to lead him past the Siq under the pretext of sacrificing a goat at Aaron's tomb, reintroducing Petra to European cartography.",
            ],
            legacy_and_modern_science=[
                "In 2016, satellite multispectral imagery and drone photogrammetry analyzed by Sarah Parcak and Christopher Tuttle revealed a previously unnoticed fifty-six-meter ceremonial platform buried in plain sight less than half a mile from Petra's center.",
                "Modern conservationists have restored ancient Nabataean flash-flood diversion channels above the Siq after recognizing that two-thousand-year-old Nabataean hydrology still outperforms modern concrete barriers.",
            ],
            metrics=[
                KeyMetric("Al-Khazneh Height", "39.1 Meters"),
                KeyMetric("Annual Rainfall", "~12 cm / Year"),
                KeyMetric("Daily Water Supply", "45 Million Liters"),
                KeyMetric("Carving Method", "Top-Down Subtractive"),
            ],
            visual_queries=["Petra Jordan Treasury Al-Khazneh", "Petra Siq canyon sandstone"],
            source_urls=["https://en.wikipedia.org/wiki/Petra"],
        ),
        SubTopicDossier(
            index=5,
            title="The Roman Colosseum: Crowd Physics, Concrete & Underground Spectacle",
            subtitle="Vespasian's Flavian Amphitheatre — structural load paths, sailor awnings, and the hypogeum machine",
            era_or_date="70 – 80 CE (Flavian Dynasty)",
            location="Rome, Italy",
            summary=(
                "Built in under ten years over Emperor Nero's private artificial lake, the Flavian Amphitheatre "
                "remains the blueprint for every modern sports stadium on Earth."
            ),
            origin_and_context=[
                "Following Nero's suicide and the chaotic Year of the Four Emperors, General Vespasian launched the amphitheatre in 70 CE as a masterstroke of political rehabilitation: draining Nero's private pleasure lake in the Domus Aurea to return Rome's valley center to the public.",
                "A reconstructed bronze-peg inscription discovered by Géza Alföldy confirmed the stadium was financed entirely ex manubiis — straight from war booty looted during Titus's 70 CE sack of Jerusalem.",
                "When Emperor Titus inaugurated the arena in 80 CE, one hundred consecutive days of gladiatorial combat, staged naval engagements, and exotic beast hunts unfolded before fifty thousand to sixty-five thousand roaring citizens seated strictly by Roman social rank.",
            ],
            engineering_and_mechanics=[
                "Measuring 189 by 156 meters on a four-center polycentric ellipse, the Colosseum solved stadium crowd crush two millennia ago: eighty numbered ground-level arches — called vomitoria — connected via independent radial staircases so fifty thousand spectators could evacuate in roughly twelve minutes.",
                "Roman architects orchestrated an ingenious vertical weight-reduction gradient: dense travertine limestone piers (100,000 cubic meters pinned with 300 tons of iron clamps) bear compressive loads at the base, lighter volcanic tuff fills mid-levels, and ultra-light porous pumice Roman concrete forms the highest vaults.",
                "Beneath the arena's sand-covered wooden floor (harena, Latin for sand to soak up blood) lay the two-story hypogeum: a labyrinth of eighty vertical shafts, hinged trapdoors, and twenty-eight bronze-capstan elevators reconstructed by Heinz-Jürgen Beste, capable of thrusting lions, leopards, and painted forest scenery simultaneously into the arena.",
                "High on the fourth-story attic corbels, a detachment of one hundred sailors from the imperial Roman fleet at Misenum rigged the velarium — a retractable linen canvas canopy tensioned across hemp ropes to shade spectators while creating a cooling central updraft.",
            ],
            human_stories_and_paradoxes=[
                "Gladiators were expensive, high-status professional athletes fed a high-carbohydrate barley-and-ash bone-strengthening diet confirmed by strontium isotope bone analysis at Ephesus — meaning fights to the death were far rarer and more regulated than Hollywood depicts.",
                "During the Middle Ages after earthquakes collapsed the southern outer ring in 1349, Roman noble families turned the arcades into fortified castles while papal builders quarried fallen travertine blocks to construct the facade of St. Peter's Basilica and Palazzo Barberini.",
            ],
            legacy_and_modern_science=[
                "Surprisingly, the gutted Colosseum became an accidental microclimate ark: botanists cataloged more than four hundred exotic plant species — some native to North Africa and Asia — sprouting from seeds carried in the fur and feed of ancient arena animals.",
                "MIT and Harvard materials scientists found that hot-mixed Roman lime clasts give Flavian concrete self-healing properties: whenever microcracks open, rainwater dissolves reactive calcium deposits that recrystallize and seal fractures automatically.",
            ],
            metrics=[
                KeyMetric("Ellipse Footprint", "189 × 156 Meters"),
                KeyMetric("Construction Speed", "9–10 Years (70–80 CE)"),
                KeyMetric("Hypogeum Lifts", "28 Capstan Elevators"),
                KeyMetric("Evacuation Time", "~12 Minutes (80 Gates)"),
            ],
            visual_queries=["Roman Colosseum Flavian Amphitheatre", "Colosseum hypogeum interior arches"],
            source_urls=["https://en.wikipedia.org/wiki/Colosseum"],
        ),
        SubTopicDossier(
            index=6,
            title="Machu Picchu: Seismic Stonecraft Above the Cloud Forest",
            subtitle="Why sixty percent of the Inca royal estate's engineering lies invisible beneath your feet",
            era_or_date="c. 1450 – 1532 CE",
            location="Cusco Region, Peruvian Andes (2,430 m)",
            summary=(
                "Perched on a narrow granite fault saddle nearly eight thousand feet above sea level above the "
                "raging Urubamba River, Machu Picchu is a triumph of geotechnical and seismic design."
            ),
            origin_and_context=[
                "Radiocarbon accelerator mass spectrometry published by Lucy Salazar and Richard Burger placed Machu Picchu's construction starting around 1420–1450 CE as a royal winter estate for Inca Emperor Pachacuti after his campaigns down the Urubamba Valley.",
                "Never discovered or plundered by Spanish conquistadors — who marched down adjacent valleys just kilometers away — the citadel was peacefully vacated in the 1530s amid smallpox epidemics and civil war, leaving its sacred Intihuatana solar stone intact.",
                "When Yale historian Hiram Bingham arrived in July 1911 guided by local farmer Melchor Arteaga and an eleven-year-old Quechua boy named Pablito Alvarez, Andean families were already farming sections of the ancient agricultural terraces.",
            ],
            engineering_and_mechanics=[
                "Paleohydrologist Kenneth Wright calculated that roughly sixty percent of total Inca construction effort at Machu Picchu lies entirely underground: before raising a single temple wall, engineers excavated deep rock-chip, gravel, and sand French-drain strata to digest two thousand millimeters of annual tropical mountain rain without a landslide.",
                "Built directly over the intersecting Machu Picchu and Huayna Picchu geological faults — which shattered regional granite into abundant quarry blocks — Inca master masons dressed stones using harder river cobble hammerstones via percussive bedding.",
                "In sacred precincts like the Temple of the Three Windows and Temple of the Sun, polygonal granite blocks interlock with zero mortar so tightly a razor blade cannot slip between joints; during Peruvian earthquakes, the beveled blocks momentarily 'dance' and resettle into alignment instead of snapping.",
                "Seven hundred stepped agricultural terraces both stabilized steep forty-degree slopes and created thermal microclimates several degrees warmer at lower elevations, preventing high-altitude frost damage to maize and coca crops.",
            ],
            human_stories_and_paradoxes=[
                "Osteological and isotopic studies of 177 individuals buried in Machu Picchu's cave cemeteries revealed neither war casualties nor slave labor, but multi-ethnic yanacona retainers brought from coast to altiplano — complete with pet Inca dogs and South American camelids.",
                "The Inca built all of Machu Picchu without iron chisels, draft horses, wheeled carts, or phonetic writing, encoding census and engineering logistics in knotted cotton khipu cords.",
            ],
            legacy_and_modern_science=[
                "Airborne LiDAR penetrating the cloud-forest canopy around Huayna Picchu recently uncovered more than a dozen ceremonial baths and agricultural aqueduct extensions concealed beneath dense bamboo and orchid growth.",
                "Modern civil engineers regard Machu Picchu's spring-collection canal — spanning 749 meters at a controlled three-percent grade feeding sixteen cascading stone fountains — as one of pre-Columbian America's finest hydraulic works.",
            ],
            metrics=[
                KeyMetric("Elevation", "2,430 m (7,972 ft)"),
                KeyMetric("Subsurface Effort", "~60% Foundation/Drains"),
                KeyMetric("Annual Rainfall", "~2,000 mm Handled"),
                KeyMetric("Stonework", "Mortarless Seismic Ashlar"),
            ],
            visual_queries=["Machu Picchu Andes mountains Peru", "Inca ashlar stonework Machu Picchu"],
            source_urls=["https://en.wikipedia.org/wiki/Machu_Picchu"],
        ),
        SubTopicDossier(
            index=7,
            title="Chichén Itzá: The Pyramid That Breathes Light and Sound",
            subtitle="Maya-Toltec archaeoastronomy, acoustic wave physics, and the subterranean cenote cosmos",
            era_or_date="c. 600 – 1200 CE",
            location="Yucatán Peninsula, Mexico",
            summary=(
                "At the center of a vast karst limestone plain devoid of surface rivers, Chichén Itzá fused "
                "naked-eye Maya astronomy, architectural acoustics, and sacred subterranean hydrology."
            ),
            origin_and_context=[
                "Named for the 'Mouth of the Well of the Itza,' the metropolis grew around massive natural sinkholes — cenotes — formed where groundwater collapsed porous limestone resting inside the outer ring of the Chicxulub asteroid impact basin.",
                "Between 800 and 1050 CE, Chichén Itzá emerged as a cosmopolitan Mesoamerican superpower blending Puuc Maya stone mosaic traditions with central Mexican feathered-serpent iconography of Kukulcán.",
                "Towering thirty meters over a leveled five-acre limestone plaza stands El Castillo, the Temple of Kukulcán — built directly encasing an older ninth-century pyramid holding a red-painted jaguar throne inlaid with seventy-three jadeite spots.",
            ],
            engineering_and_mechanics=[
                "El Castillo is a physical stone calendar: four staircases of ninety-one steps each plus the shared summit temple platform total exactly 365 steps for the Haab' solar year, while nine stepped terraces split by staircases create eighteen sections matching the eighteen twenty-day Maya calendar months.",
                "Rotated twenty-one degrees clockwise off cardinal axes, the northwest corner terraces intercept late-afternoon sunbeams on the March and September equinoxes, projecting seven undulating isosceles triangles of light down the northern balustrade to unite with a carved stone serpent head at the base.",
                "Acoustical physicist David Lubman demonstrated that clapping your hands in front of El Castillo's steep limestone stairs produces not a flat slap-echo, but a frequency-swept 'chirped' diffraction echo descending from roughly 1,400 to 900 hertz — uncannily mimicking the sacred call of the resplendent quetzal bird.",
                "Across the Great Ball Court — the largest in Mesoamerica at 168 by 70 meters — parallel nine-meter vertical stone walls act as an acoustic waveguide allowing a conversational whisper at the North Temple to be heard clearly 140 meters away at the South Temple.",
            ],
            human_stories_and_paradoxes=[
                "Dredging of the sixty-meter-wide Sacred Cenote yielded gold bells from Panama, turquoise from New Mexico, preserved copal incense, Maya wooden textiles, and skeletal remains confirming long-distance pilgrimage networks spanning three thousand kilometers.",
                "Unlike Classic Maya city-states ruled by single divine kings, Chichén Itzá pioneered a council-based multepal governance model that kept the city thriving centuries after southern lowland capitals like Tikal and Palenque collapsed during ninth-century megadroughts.",
            ],
            legacy_and_modern_science=[
                "In 2015 and 2018, Mexican geophysicists led by René Chávez used three-dimensional electrical resistivity tomography beneath El Castillo and discovered El Castillo sits directly atop an undiscovered twenty-meter-deep subterranean cenote cavern — placing the pyramid literally at the axis mundi over living water.",
                "Archaeologists warn that eroding limestone edges by even a few centimeters could permanently silence El Castillo's thousand-year-old quetzal acoustic diffraction.",
            ],
            metrics=[
                KeyMetric("Total Calendar Steps", "365 (4 × 91 + 1)"),
                KeyMetric("Pyramid Nested Shells", "3 Concentric Pyramids"),
                KeyMetric("Acoustic Echo", "900–1400 Hz Quetzal Chirp"),
                KeyMetric("Hidden Base", "Subterranean Cenote Cave"),
            ],
            visual_queries=["Chichen Itza El Castillo pyramid", "Temple of Kukulkan Mexico"],
            source_urls=["https://en.wikipedia.org/wiki/Chichen_Itza"],
        ),
        SubTopicDossier(
            index=8,
            title="The Taj Mahal: Optical Illusions, River Foundations & Mughal Symmetry",
            subtitle="How Ustad Ahmad Lahauri floated 12,000 tons of marble beside a shifting monsoon river",
            era_or_date="1632 – 1653 CE",
            location="Agra, Uttar Pradesh, India",
            summary=(
                "Commissioned by Mughal Emperor Shah Jahan for his beloved wife Mumtaz Mahal, the Taj Mahal "
                "marries poetic grief with daring seventeenth-century structural and geotechnical engineering."
            ),
            origin_and_context=[
                "In June 1631, Empress Mumtaz Mahal — Shah Jahan's trusted political confidante who accompanied him on military campaigns — died at Burhanpur giving birth to their fourteenth child; contemporary court chronicles record the emperor's beard turned silver within months.",
                "Over twenty-one years, chief imperial architect Ustad Ahmad Lahauri mobilized twenty thousand stonecutters, calligraphers, dome-builders from Bukhara and Constantinople, and lapidary masters across Eurasia alongside one thousand elephants hauling translucent white marble three hundred kilometers from Makrana in Rajasthan.",
                "Unlike traditional Mughal tombs placed at the center of a cross-axial Charbagh garden, Lahauri audaciously anchored the mausoleum at the extreme northern edge right over the Yamuna River bank so the sky alone — never dark foliage — would frame its seventy-three-meter marble dome.",
            ],
            engineering_and_mechanics=[
                "Building a colossal stone monument on saturated, shifting alluvial riverbank mud should have caused catastrophic differential settlement; Mughal engineers solved this by sinking dozens of brick-lined caisson wells below the water table, filling them with rubble mortar, and capping them on timber cribs of rot-proof sal and ebony wood that stay rock-hard as long as they remain submerged.",
                "The iconic bulbous outer 'onion' dome rises thirty-five meters above a much lower inner structural dome, creating a hollow double-shell acoustic chamber whose twenty-eight-second reverberation time turns a single flute note or whispered prayer into a shimmering choral chord.",
                "All four forty-meter corner minarets were intentionally constructed with a subtle outward tilt of roughly two degrees away from the central plinth: if a major Himalayan earthquake ever toppled them, they would fall harmlessly into the outer courtyard rather than smashing the royal cenotaph.",
                "Calligrapher Amanat Khan engineered perspective-corrected Quranic jasper inlays around the thirty-meter pishtaq archway, progressively enlarging letter heights toward the top so verses appear perfectly uniform in size to a viewer standing at ground level.",
            ],
            human_stories_and_paradoxes=[
                "No historical record supports the nineteenth-century colonial rumor that Shah Jahan amputated artisans' hands or planned a mirror-image Black Taj across the Yamuna; what darkens across the river are ruins of the Moonlight Garden — Mehtab Bagh — whose octagonal pool caught the white Taj's reflection under full moons.",
                "In a Shakespearean twist of dynastic tragedy, Shah Jahan's own son Aurangzeb deposed him in 1658 and imprisoned the aging emperor inside Agra Fort, leaving him to gaze eight hundred meters downriver at his wife's luminous tomb for the final eight years of his life.",
            ],
            legacy_and_modern_science=[
                "Under changing daylight, Makrana's low-porosity calcitic marble acts as a natural optical diffuser — shifting from pearlescent rose at dawn to blinding alabaster at noon and cool silver-blue by moonlight — because twenty-eight varieties of semi-precious carnelian, lapis lazuli, turquoise, and jade are inlaid via seamless pietra dura.",
                "Today environmental engineers monitor both industrial particulate yellowing — treated gently with traditional Fuller's earth clay packs (multani mitti) — and Yamuna groundwater levels vital to keeping the seventeenth-century timber well foundations saturated.",
            ],
            metrics=[
                KeyMetric("Central Finial Height", "73 Meters (240 ft)"),
                KeyMetric("Artisan Workforce", "~20,000 Specialists"),
                KeyMetric("Minaret Safety Tilt", "~2° Outward Seismic Lean"),
                KeyMetric("Dome Reverberation", "~28 Seconds"),
            ],
            visual_queries=["Taj Mahal sunrise reflection Agra", "Taj Mahal marble pietra dura architecture"],
            source_urls=["https://en.wikipedia.org/wiki/Taj_Mahal"],
        ),
        SubTopicDossier(
            index=9,
            title="Christ the Redeemer & The Grand Synthesis of Human Monumentality",
            subtitle="Twentieth-century Art Deco aerodynamics atop Corcovado — and why wonders endure",
            era_or_date="1922 – 1931 CE & Synthesis",
            location="Mount Corcovado, Rio de Janeiro, Brazil",
            summary=(
                "Crowning a 710-meter granite pinnacle battered by Atlantic gales and tropical lightning, "
                "Christ the Redeemer brings ancient monumental ambition into the reinforced-concrete modern era."
            ),
            origin_and_context=[
                "Conceived for Brazil's 1922 independence centennial and funded entirely through nationwide citizen donations collected by Rio's archdiocese, the monument replaced an early sketch of a globe-carrying statue with Brazilian engineer Heitor da Silva Costa's bold cruciform silhouette visible across Guanabara Bay.",
                "Truly transnational in execution, French-Polish sculptor Paul Landowski modeled the four-meter hands and torso in clay outside Paris, Romanian sculptor Gheorghe Leonida carved the serene Art Deco face, and French structural pioneer Albert Caquot engineered the internal wind-frame.",
                "Workers hauled every sack of cement, steel rebar, and scaffolding beam up the steep Corcovado cog railway before hanging precariously over seven-hundred-meter sheer drops — remarkably completing nine years of high-altitude construction with zero recorded worker fatalities.",
            ],
            engineering_and_mechanics=[
                "With twenty-eight-meter outstretched arms acting like a giant aircraft wing in coastal hurricane gusts, Albert Caquot designed four internal reinforced-concrete columns tied to horizontal trusses capable of resisting winds exceeding 250 kilometers per hour.",
                "During a 1923 trip to fountain squares in Sweden, Silva Costa noticed pale green-gray soapstone (steatite) resisted thermal cracking, acid rain, and salt spray; thousands of Rio volunteers glued roughly six million triangular three-centimeter soapstone tiles onto linen sheets — often writing hidden prayers on the backs before masons pressed them into mortar.",
                "Standing at the highest conductive peak over tropical Rio, Christ the Redeemer gets struck by lightning four to six times every single year, protected by a crown-of-thorns Faraday array of platinum-tipped lightning rods running down the arms into Corcovado's granite core.",
                "Looking back across all wonders — from Giza's limestone mass to Petra's rock-cut hydraulics, Rome's self-healing concrete, Machu Picchu's dancing granite, Chichén Itzá's acoustic steps, Agra's river caissons, and Rio's reinforced concrete — every era pushed materials science past known limits.",
            ],
            human_stories_and_paradoxes=[
                "On October 12, 1931, radio pioneer Guglielmo Marconi attempted to switch on Christ the Redeemer's floodlights wirelessly via shortwave radio signal transmitted 9,200 kilometers from Rome — though stormy atmospheric static forced technicians on Corcovado to flip the switch manually at the decisive second.",
                "What unites a Bronze Age pharaoh, Nabataean desert merchants, Inca hydraulic planners, a grieving Mughal emperor, and twentieth-century Brazilian citizens is refusal to accept human impermanence.",
            ],
            legacy_and_modern_science=[
                "In recent decades, laser LiDAR point-cloud twins with millimeter accuracy have created permanent digital blueprints of all surviving wonders, ensuring that even if earthquakes or climate extremes strike, their exact geometry can never be lost the way Alexandria's Pharos or Rhodes's Colossus vanished.",
                "Ultimately, the Seven Wonders are not monuments to stone or concrete — they are proof of what happens when human imagination refuses to bow to gravity, geography, or time.",
            ],
            metrics=[
                KeyMetric("Corcovado Peak", "710 m Above Rio"),
                KeyMetric("Arm Wingspan", "28 Meters (92 ft)"),
                KeyMetric("Soapstone Mosaic", "~6 Million Tiles"),
                KeyMetric("Lightning Strikes", "4–6 Direct Hits / Yr"),
            ],
            visual_queries=["Christ the Redeemer Rio Corcovado", "Seven Wonders World synthesis"],
            source_urls=["https://en.wikipedia.org/wiki/Christ_the_Redeemer_(statue)"],
        ),
    ]

    return ResearchDossier(
        topic=topic_title,
        overview=(
            "A comprehensive historical, structural, and archaeological investigation spanning 4,500 years "
            "from the original Hellenistic Seven Wonders of the Ancient World to the surviving Great Pyramid "
            "of Giza and the complete modern New Seven Wonders canon."
        ),
        thesis_statement=(
            "Behind the postcard silhouettes of the Seven Wonders lies a hidden history of extreme structural "
            "engineering — sticky-rice mortar, seismic dancing stone, pressurized desert pipelines, acoustic "
            "staircases, river-well foundations, and human obsession with outlasting time."
        ),
        cold_open_hook_facts=[
            "Greek writers named seven wonders. Six are gone. Khufu's Great Pyramid still rises above Giza, built about four thousand five hundred years ago.",
            "Hidden inside today's surviving wonders are secret corridors detected by cosmic rays from deep space, desert water networks that carried twelve million gallons a day, staircases engineered to chirp like tropical birds, and four-hundred-ton marble domes floating on submerged ebony wood.",
        ],
        synthesis_comparisons=[
            "Ancient bronze giants like the Colossus of Rhodes collapsed within a lifetime because metal corrodes and snaps under tension, while Giza, Machu Picchu, and Rome harnessed pure stone compression and seismic flexibility.",
            "Across every continent — Egypt, China, Jordan, Italy, Peru, Mexico, India, and Brazil — monumental architecture emerged whenever hydrology, material chemistry, and collective purpose converged.",
        ],
        subtopics=subtopics,
        sources_consulted=[
            "https://en.wikipedia.org/wiki/Wonders_of_the_World",
            "https://en.wikipedia.org/wiki/Seven_Wonders_of_the_Ancient_World",
            "https://en.wikipedia.org/wiki/New_7_Wonders_of_the_World",
            "https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza",
            "https://en.wikipedia.org/wiki/Great_Wall_of_China",
            "https://en.wikipedia.org/wiki/Petra",
            "https://en.wikipedia.org/wiki/Colosseum",
            "https://en.wikipedia.org/wiki/Machu_Picchu",
            "https://en.wikipedia.org/wiki/Chichen_Itza",
            "https://en.wikipedia.org/wiki/Taj_Mahal",
            "https://en.wikipedia.org/wiki/Christ_the_Redeemer_(statue)",
        ],
        live_web_enriched=False,
    )
