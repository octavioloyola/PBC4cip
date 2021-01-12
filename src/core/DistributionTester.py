def AlwaysTrue(distribution, model, classFeature):
    return True


def PureNodeStopCondition(distribution, model, classFeature):
    stop = (max(distribution) == sum(distribution))
    return stop
