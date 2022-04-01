from plotnine import *


def plot_retirement_assets(simulation, standard_of_living):
    simulation = simulation.query('retired==True').dropna()
    under_funded = simulation.groupby('simulation').filter(lambda df: df.spend.min() < standard_of_living)

    return (
        simulation
        .groupby('age').assets
        .quantile([.025, .5, .975])
        .unstack(level=1)
        .rename(columns={.025: 'low', 0.5: 'mid', .975: 'hi'})
        .pipe(lambda x: x / 1e6)
        .reset_index()
        .pipe(ggplot)
        + aes(x='age', y='mid', ymin='low', ymax='hi')
        + geom_ribbon(fill='#ccc')
        + geom_line(alpha=1, color='k')
        + geom_line(aes(x='age', y='assets / 1e6', group='simulation'), alpha=0.06, data=simulation,
                    inherit_aes=False)
        + geom_line(aes(x='age', y='assets / 1e6', group='simulation'), alpha=0.8, color='r', data=under_funded,
                    inherit_aes=False)
        + labs(x='Age', y='Portfolio ($M)')
        + theme_538()
        + theme(
            dpi=120
        )
    )


def plot_retirement_age(simulation, standard_of_living):

    simulation = simulation.query('retired==True').dropna()
    under_funded = simulation.groupby('simulation').filter(lambda df: df.spend.min() < standard_of_living)
    retirement_age = simulation.groupby('simulation').age.min().reset_index()
    return (
        retirement_age
        .pipe(ggplot)
        + aes(x='age')
        + geom_histogram(bins=10)
        + geom_histogram(bins=10, data=under_funded.groupby('simulation').age.min().reset_index(), fill='r')
        + labs(
            x='Retirement Age',
            y='Number of Simulations'
        )
        + theme_538()
        + theme(
            dpi=120
        )
    )


def plot_spend_timeline(simulation, standard_of_living):
    under_funded = simulation.dropna().groupby('simulation').filter(lambda df: df.spend.min() < standard_of_living)

    return (
        simulation.dropna()
        .groupby('age').spend
        .quantile([.025, .5, .975])
        .unstack(level=1)
        .rename(columns={.025: 'low', 0.5: 'mid', .975: 'hi'})
        .pipe(lambda x: x / 1e3)
        .reset_index()
        .pipe(ggplot)
        + aes(x='age', y='mid', ymin='low', ymax='hi')
        + geom_ribbon(fill='#ccc')
        + geom_line(aes(x='age', y='spend / 1e3', group='simulation'), alpha=0.06, data=simulation,
                    inherit_aes=False)
        + geom_line(aes(x='age', y='spend / 1e3', group='simulation'), alpha=0.8, color='r', data=under_funded,
                    inherit_aes=False)
        + labs(x='Age', y='Yearly Spend ($K)')
        + theme_538()
        + theme(
           dpi=120
        )
    )